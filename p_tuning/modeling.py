import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical, kl

from os.path import join
from transformers import AutoTokenizer

from p_tuning.models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from data_utils.dataset import load_file
from p_tuning.prompt_encoder import PromptEncoder

from p_tuning.adapter import adapter

import nltk
from nltk.corpus import stopwords

try:
    print(stopwords.words('english'))
except:
    nltk.download('stopwords')

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device
        # loss hyperparameter
        self.me_loss_topk = self.args.me_loss_topk
        self.lambda_me = self.args.lambda_me # 0.2
        self.lambda_kl = self.args.lambda_kl # 0.2
        self.lambda_aug = self.args.lambda_aug # 0.5

        # load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join(self.args.data_dir, 'relations.jsonl'))
        )
        self.paratrex_templates = [
            {args.relation_id: d['template']}
            for d in load_file(join(self.args.data_dir, 'ParaTrex', args.relation_id+'.jsonl')) if d['inhouse_split']=='train'
        ]
        self.pararel_templates = [
            {args.relation_id: d['pattern']}
            for d in load_file(join(self.args.data_dir, 'pararel', args.relation_id+'.jsonl')) 
        ]

        # load tokenizer
        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src, use_fast=False)

        # load pre-trained model
        if 'megatron' in self.args.model_name and self.args.use_lm_finetune:
            raise RuntimeError(
                "Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        self.model = create_model(self.args)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        if args.use_adaptor:
            adapter(self.model, args.adaptor_hidden_dim, self.device)
                
        self.embeddings = get_embedding_layer(self.args, self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(
            self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[
            self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(
            self.template, self.hidden_size, self.tokenizer, args).to(self.device)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

        self.softmax = nn.Softmax(dim=-1)
        eng_stopwords = stopwords.words('english')
        self.eng_stopwords_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(eng_stopwords)).to(self.device)
        self.output_embedding = self.model.get_output_embeddings().weight

        self.neg_sample_num = 64
        self.cre = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Linear(self.hidden_size, 2).to(self.device)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)
                              ] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        # For using handcraft prompts
        if self.args.use_original_template:
            return raw_embeds

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape(
            (bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i],
                           :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None, mask=None, random=False, aug=None):
        # For using handcraft prompts
        if self.args.use_original_template:
            if random is True:
                rand_idx = torch.randint(0, len(self.paratrex_templates), (1,))
                rand_relation_templates = self.paratrex_templates[rand_idx][self.args.relation_id]
                query = rand_relation_templates.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                return self.tokenizer(' ' + query)['input_ids']
            if aug is not None:
                relation_templates = self.relation_templates[self.args.relation_id]
                if aug == 'its_true':
                    relation_templates = 'It is true that' + relation_templates[0].lower() + relation_templates[1:]
                elif aug == 'its_false':
                    relation_templates = 'It is false that' + relation_templates[0].lower() + relation_templates[1:]
                else:
                    raise ValueError('must be either its_true or its_false')
                query = relation_templates.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                return self.tokenizer(' ' + query)['input_ids']

            special_tokens = ['[X]', '[Y]']
            self.tokenizer.add_tokens(special_tokens)
            if mask == 'subject':
                relation_templates = self.relation_templates[self.args.relation_id]
                subject_length = len(self.tokenizer.tokenize(' '+x_h))
                query = relation_templates.replace('[X] ', (self.tokenizer.mask_token+' ')*subject_length).replace(
                    '[Y]', self.tokenizer.mask_token)
            elif mask == 'object':
                relation_templates = self.relation_templates[self.args.relation_id]
                query = relation_templates.replace('[X]', self.tokenizer.mask_token).replace(
                    '[Y]', 'a') 
                # here we use a as object since only the location of the first token of subject matters because queries_maksed_obj is only used for getting the position of subject mask, not passing to lm
            elif mask == 'sub_obj':
                relation_templates = self.relation_templates[self.args.relation_id]
                query = relation_templates.replace('[X]', self.tokenizer.mask_token).replace(
                    '[Y]', self.tokenizer.mask_token) # here we use a as object since only the location of the first token of subject matters
            else:
                # elif mask == None
                relation_templates = self.relation_templates[self.args.relation_id]
                query = relation_templates.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
            return self.tokenizer(' ' + query)['input_ids']
        # For P-tuning
        # BERT-style model
        if mask == 'subject':
            if aug == 'its_true':
                return [[self.tokenizer.cls_token_id]  # [CLS]
                + self.tokenizer.encode('It is true that', add_special_tokens=False)
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                + [self.tokenizer.mask_token_id] * len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h)))# [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]
            elif aug == 'its_false':
                return [[self.tokenizer.cls_token_id]  # [CLS]
                + self.tokenizer.encode('It is false that', add_special_tokens=False)
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                + [self.tokenizer.mask_token_id] * len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h)))# [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

            return [[self.tokenizer.cls_token_id]  # [CLS]
            + prompt_tokens * self.template[0]
            + [self.tokenizer.mask_token_id]  # head entity
            + prompt_tokens * self.template[1]
            + [self.tokenizer.mask_token_id] * len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h)))# [MASK] (tail entity)
            + (prompt_tokens * self.template[2] if self.template[
                                                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
            + [self.tokenizer.sep_token_id]
            ]
        elif mask == 'sub_obj':
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    # [MASK] (tail entity)
                    + [self.tokenizer.mask_token_id]
                    + (prompt_tokens * self.template[2] if self.template[
                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        elif mask == 'object':
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    # [MASK] (tail entity)
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' a'))
                    + (prompt_tokens * self.template[2] if self.template[
                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        else:
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    # [MASK] (tail entity)
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))
                    + (prompt_tokens * self.template[2] if self.template[
                        2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]

    def forward(self, x_hs, x_ts, return_candidates=False, mask_mode=None):
        
        bz = len(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(
            x_hs[i], prompt_tokens, mask=None)).squeeze(0) for i in range(bz)]
        raw_len = len(queries)

        # counterfactual data
        queries_maksed_sub = [torch.LongTensor(self.get_query(
            x_hs[i], prompt_tokens, mask='subject')).squeeze(0) for i in range(bz)]
        ctf_len = raw_len+len(queries_maksed_sub)
        queries += queries_maksed_sub
        # paraphrase data
        if self.args.paraphrase_loss is True:
            queries_para = [torch.LongTensor(self.get_query(
                x_hs[i], prompt_tokens, mask=None, random=True)).squeeze(0) for i in range(bz)]
            para_len = len(queries)+len(queries_para)
            queries += queries_para
            queries_para2 = [torch.LongTensor(self.get_query(
                x_hs[i], prompt_tokens, mask=None, random=True)).squeeze(0) for i in range(bz)]
            para_len2 = para_len+len(queries_para2)
            queries += queries_para2
        # augmentation data
        if self.args.aug_loss is True:
            queries_aug = [torch.LongTensor(self.get_query(
                x_hs[i], prompt_tokens, mask=None, aug='its_true')).squeeze(0) for i in range(bz)]
            aug_len = len(queries)+len(queries_aug)
            queries += queries_aug
            queries_aug2 = [torch.LongTensor(self.get_query(
                x_hs[i], prompt_tokens, mask=None, aug='its_false')).squeeze(0) for i in range(bz)]
            aug_len2 = aug_len+len(queries_aug2)
            queries += queries_aug2
        

        # queries = [torch.LongTensor(self.get_query(
        #     x_hs[i], prompt_tokens, mask=None)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(
            queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask = (queries != self.pad_token_id)
        # get embedded input
        inputs_embeds = self.embed_input(queries)


        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        

        def bert_out():
            label_mask = (queries[:raw_len, :] == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
                1).to(self.device)  # bz * 1
            labels = torch.empty_like(queries[:raw_len, :]).fill_(-100).long().to(self.device)  # bz * seq_len
            labels = labels.scatter_(1, label_mask, label_ids)

            outputs = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                output_hidden_states=True,
                                output_attentions=True)
            full_logits = outputs.logits #(bs, seq_len, vocab_size)
            # split logits
            logits = full_logits[:raw_len, :, :]
            counter_logits = full_logits[raw_len:ctf_len, :, :]
            # get specific preds
            counter_pred_logits, counter_pred_ids = torch.sort(counter_logits, dim=2, descending=True)
            if self.args.aug_loss is True:
                aug_logits_true = full_logits[(aug_len-bz):aug_len, :, :]
                aug_logits_false = full_logits[aug_len:aug_len2, :, :]
                logits = (logits + self.lambda_aug*(aug_logits_true - aug_logits_false))/2
            sorted_logits, pred_ids = torch.sort(logits, dim=2, descending=True)

            if self.args.paraphrase_loss is True:
                kldiv = []
                for i in range(self.args.paraphrase_times):
                    para_logits = full_logits[para_len[i]:para_len[i+1], :, :]
                    label_mask_para = (queries[para_len[i]:para_len[i+1], :] == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  # bz * 
                    probs_aug = Categorical(probs=F.softmax(para_logits[torch.arange(bz), label_mask_para[torch.arange(bz), 0], :], dim=1))
                    kl_div = torch.sum(torch.distributions.kl.kl_divergence(raw_prob, probs_aug))
                    kldiv.append(kl_div.item())
                kl_div_loss = sum(kldiv)/len(kldiv)

            # subj_obj masked case:
            queries_maksed_obj = torch.stack([torch.LongTensor(self.get_query(
            x_hs[i], prompt_tokens, mask='object')).squeeze(0) for i in range(bz)])
            # queries_maksed_obj is only used for getting the position of subject mask, not passing to lm
            label_mask_sub = (queries_maksed_obj == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
                    1).to(self.device)  # bz * 1
            subj_obj_maksed_queries = [torch.LongTensor(self.get_query(
                x_hs[i], prompt_tokens, mask='sub_obj')
                ).squeeze(0) for i in range(bz)]
            subj_obj_maksed_queries = pad_sequence(
                subj_obj_maksed_queries, True, padding_value=self.pad_token_id).long().to(self.device)
            subj_obj_maksed_attention_mask = (subj_obj_maksed_queries != self.pad_token_id)
            subj_obj_maksed_inputs_embeds = self.embed_input(subj_obj_maksed_queries)
            subj_obj_maksed_outputs = self.model(
                                inputs_embeds=subj_obj_maksed_inputs_embeds.to(self.device),
                                attention_mask=subj_obj_maksed_attention_mask.to(self.device).bool(),
                                output_hidden_states=True,
                                output_attentions=True
                                )
            subj_obj_maksed_logits = subj_obj_maksed_outputs.logits #(bs, seq_len, vocab_size)
            counter_sub_logits, counter_sub_pred_ids = torch.sort(subj_obj_maksed_logits, dim=2, descending=True)

            # mlm loss
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            
            me_loss = 0
            hit1 = 0
            counter_hit1 = 0
            mrr = 0
            kld = 0
            loss_entropy = 0
            top10 = []

            if not self.training:
                # calculate kl divergence
                raw_prob = Categorical(probs=F.softmax(logits[torch.arange(bz), label_mask[torch.arange(bz), 0], :], dim=1))
                counter_prob = Categorical(probs=F.softmax(counter_logits[torch.arange(bz), label_mask[torch.arange(bz), 0], :], dim=1))
                kld += (torch.sum(torch.distributions.kl.kl_divergence(raw_prob, counter_prob)).item()+torch.sum(torch.distributions.kl.kl_divergence(counter_prob, raw_prob)).item())/2
            
            for i in range(bz):
                # masked subj, max the entropy of subj pred 
                counter_topk_sub_candidates = counter_sub_pred_ids[i, label_mask_sub[i, 0], :self.me_loss_topk]
                entropy_mask = torch.zeros(counter_sub_pred_ids[i, label_mask_sub[i, 0]].shape).to(self.device)
                mask = torch.logical_not(torch.isin(counter_topk_sub_candidates, self.eng_stopwords_ids))
                entropy_mask[:self.me_loss_topk][mask] = 1
                log_probs = torch.nn.functional.log_softmax(counter_sub_logits[i, label_mask_sub[i, 0]], dim=-1)
                probs = self.softmax(counter_sub_logits[i, label_mask_sub[i, 0]])
                me_loss_obj = torch.sum(probs.mul(log_probs).mul(entropy_mask))
                # masked subj, max the entropy of obj pred 
                counter_topk_candidates = counter_pred_ids[i, label_mask[i, 0], :self.me_loss_topk]
                entropy_mask = torch.zeros(counter_pred_ids[i, label_mask[i, 0]].shape).to(self.device)
                mask = torch.logical_not(torch.isin(counter_topk_candidates, self.eng_stopwords_ids))
                entropy_mask[:self.me_loss_topk][mask] = 1
                # entropy_mask[:300] = 1
                log_probs = torch.nn.functional.log_softmax(counter_logits[i, label_mask[i, 0]], dim=-1)
                probs = self.softmax(counter_logits[i, label_mask[i, 0]])
                me_loss_sub = torch.sum(probs.mul(log_probs).mul(entropy_mask))

                me_loss += (me_loss_obj + me_loss_sub)/2
                
                if not self.training:
                    # pred entropy
                    # candidates = pred_ids[i, label_mask[i, 0], :10]
                    counter_candidates = counter_pred_ids[i, label_mask[i, 0], :10]
                    # mask entity except top 10 logits
                    counter_top10_logits = counter_logits[i, label_mask[i, 0], counter_candidates]
                    entropy = Categorical(self.softmax(torch.tensor(counter_top10_logits))).entropy().item()
                    loss_entropy += entropy
                    # MRR
                    pred_ids_top_100 = pred_ids[i, label_mask[i, 0], :100]
                    rr = (pred_ids_top_100 == label_ids[i, 0]).nonzero(as_tuple=True)[0].cpu().numpy()
                    if len(rr) != 0:
                        mrr += 1 / (rr[0]+1)
                    # counter hit1
                    counter_pred_seq = counter_pred_ids[i, label_mask[i, 0]].tolist()
                    for pred in counter_pred_seq:
                        if pred in self.allowed_vocab_ids:
                            break
                    if pred == label_ids[i, 0]:
                        counter_hit1 += 1

                # hit
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        break
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates and (not self.training):
                top10_pred = []
                top10_logit = []
                top10_hit = []
                ans_ids = []
                for i in range(bz):
                    top10_pred.append([])
                    top10_logit.append([])
                    pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                    top10_logits = sorted_logits[i, label_mask[i, 0]].tolist()
                    for idx, pred in enumerate(pred_seq):
                        if pred in self.allowed_vocab_ids:
                            top10_pred[-1].append(pred)
                            top10_logit[-1].append(top10_logits[idx])
                            if len(top10_pred[-1]) >= 1000:
                                break
                    pred = top10_pred[-1][0]
                    ans_ids.append(label_ids[i, 0])
                    if pred == label_ids[i, 0]:
                        top10_hit.append(True) 
                    else:
                        top10_hit.append(False) 
                ct_top10_pred = []
                ct_top10_logit = []
                for i in range(bz):
                    ct_top10_pred.append([])
                    ct_top10_logit.append([])
                    ct_pred_seq = counter_pred_ids[i, label_mask[i, 0]].tolist()
                    ct_top10_logits = counter_pred_logits[i, label_mask[i, 0]].tolist()
                    for idx, pred in enumerate(ct_pred_seq):
                        if pred in self.allowed_vocab_ids:
                            ct_top10_pred[-1].append(pred)
                            ct_top10_logit[-1].append(ct_top10_logits[idx])
                            if len(ct_top10_pred[-1]) >= 1000:
                                break
                top10 = [top10_pred, top10_logit, top10_hit, ct_top10_pred, ct_top10_logit, torch.tensor(ans_ids)]
                return loss, kld, hit1, top10, mrr, loss_entropy, counter_hit1

            if not self.training:
                return loss, kld, hit1, mrr, loss_entropy, counter_hit1
            
            if self.args.paraphrase_loss is True:
                return loss + self.lambda_me*me_loss + self.lambda_kl*kl_div_loss, hit1
            else:
                return loss + self.lambda_me*me_loss, hit1


        if 'bert' in self.args.model_name:
            return bert_out()
        else:
            raise NotImplementedError()
    
    def cal_consistency(self, x_hs, x_ts):
        raw_template = self.relation_templates
        bz = len(x_hs)
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(bz, -1).to(self.device)
        
        # if mode == 'ParaTrex':
        id_top1_preds = self.cal_pred_per_template(x_hs, x_ts, label_ids).unsqueeze(1) # bz, 1
        pr_top1_preds = self.cal_pred_per_template(x_hs, x_ts, label_ids).unsqueeze(1)
        for i in self.paratrex_templates:
            self.relation_templates = i
            top1_pred = self.cal_pred_per_template(x_hs, x_ts, label_ids).unsqueeze(1)
            id_top1_preds = torch.cat([id_top1_preds, top1_pred], dim=1) 
        for i in self.pararel_templates:
            self.relation_templates = i
            top1_pred = self.cal_pred_per_template(x_hs, x_ts, label_ids).unsqueeze(1)
            pr_top1_preds = torch.cat([pr_top1_preds, top1_pred], dim=1) 
        # top1_pred: bz, n_relations
        self.relation_templates = raw_template
        return id_top1_preds, pr_top1_preds, label_ids

    def cal_pred_per_template(self, x_hs, x_ts, label_ids):        
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(
            x_hs[i], prompt_tokens, mask=None)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(
            queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask = (queries != self.pad_token_id)
        # get embedded input
        inputs_embeds = self.embed_input(queries)
        # construct label ids

        label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
            1).to(self.device)  # bz * 1
        labels = torch.empty_like(
            queries).fill_(-100).long().to(self.device)  # bz * seq_len
        labels = labels.scatter_(1, label_mask, label_ids)
        outputs = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                labels=labels.to(self.device), output_attentions=True)
        loss, logits = outputs[:2] #(bs, seq_len, vocab_size)
        pred_logits, pred_ids = torch.sort(logits, dim=2, descending=True)
        candidates = torch.cat([pred_ids[i, label_mask[i, 0], 0].unsqueeze(0) for i in range(bz)]) # bz
        pred_ids = torch.argsort(logits, dim=2, descending=True)
        return candidates
    