import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

import time
from os.path import join, abspath, dirname

from data_utils.dataset import load_file, LAMADataset

from data_utils.vocab import init_vocab
from p_tuning.modeling import PTuneForLAMA

from sklearn.metrics import f1_score

SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large',]


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()
    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default=None)
    parser.add_argument("--model_name", type=str,
                        default='bert-base-cased', choices=SUPPORT_MODELS)
    parser.add_argument("--run_name", type=str, default='default')
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--template", type=str, default="(3, 3, 3)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--pt_lr", type=float, default=1e-5)
    parser.add_argument("--ad_lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=50,
                        help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--me_loss_topk", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--vocab_strategy", type=str,
                        default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str,
                        default=join(abspath(dirname(__file__)), 'data'))
    parser.add_argument("--out_dir", type=str,
                        default=join(abspath(dirname(__file__)), 'output125/'))
    parser.add_argument("--load_checkpoint_dir", type=str,
                        default=None)                    
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str,
                        default=join(abspath(dirname(__file__)), '../checkpoints'))

    # Adaptor
    parser.add_argument("--use_adaptor", type=bool, default=False)
    parser.add_argument("--adaptor_hidden_dim", type=int, default=256)

    # Prompt-mask
    parser.add_argument("--prompt_mask_mode", type=str,
                        default=None, help='distribution, data_aug or None')
    parser.add_argument("--data_mask", type=str,
                        default=None, help='all, subject or template')
    
    # Loss configuration
    parser.add_argument("--paraphrase_loss", type=bool, default=False)
    parser.add_argument("--aug_loss", type=bool, default=False)
    parser.add_argument("--lambda_me", type=float, default=0.2)
    parser.add_argument("--lambda_kl", type=float, default=0.2)
    parser.add_argument("--lambda_aug", type=float, default=0.5)
    parser.add_argument("--paraphrase_times", type=int, default=2)

    
    args = parser.parse_args()
    
    args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(
        args.template) is not tuple else args.template
    print('Device:', args.device)
    assert type(args.template) is tuple

    set_seed(args)

    if 'roberta' in args.model_name:
        args.me_loss_topk = 300
    elif 'bert' in args.model_name:
        args.me_loss_topk = 300

    return args

def cal_consistency(top1_preds, label_ids):
    bz, n_rel = top1_preds.size()
    consist = torch.sum(top1_preds[:, 0].unsqueeze(1) == top1_preds[:, 1:], 1) / (n_rel-1) # bz

    n_all_pairs = n_rel*(n_rel - 1)/2
    consist_permutation = (torch.sum(top1_preds.unsqueeze(2) == top1_preds.unsqueeze(1), dim=-1)-1)/2
    all_consist = torch.sum(consist_permutation, dim=-1)/n_all_pairs # bz

    correct_pairs = torch.sum(top1_preds == label_ids.reshape(bz,1), dim=-1)
    acc_consist = correct_pairs * (correct_pairs-1) / 2 
    acc_consist = acc_consist/n_all_pairs

    avg_p_hit_per_x = torch.mean((top1_preds == label_ids.reshape(bz,1)).float(), dim=-1)

    return (consist, all_consist, acc_consist, avg_p_hit_per_x)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = 'cuda:0' 
        else:
            self.device = 'cpu'

        # load tokenizer
        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src, use_fast=False)
        self.omitted_relation = ['P527', 'P31', 'P166', 'P69', 'P47', 'P463', 'P101', 'P1923', 'P106', 'P530', 'P27', 'P178', 'P1412', 'P108', 'P39', 'P937', 'P1303', 'P190', 'P1001'] # N-M relations
        self.best_ckpt = None

        # load datasets and dataloaders
        init_vocab(args)
        self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()

        self.train_data = load_file(
            join(self.args.data_dir, self.data_path_pre + 'train' + self.data_path_post))
        self.dev_data = load_file(
            join(self.args.data_dir, self.data_path_pre + 'dev' + self.data_path_post))
        self.test_data = load_file(
            join(self.args.data_dir, self.data_path_pre + 'test' + self.data_path_post))

        self.test_set = LAMADataset(
            'test', self.test_data, self.tokenizer, self.args)
        self.train_set = LAMADataset(
            'train', self.train_data, self.tokenizer, self.args)
        self.dev_set = LAMADataset(
            'dev', self.dev_data, self.tokenizer, self.args)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(
            self.train_set, batch_size=8, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8)
        self.test_loader = DataLoader(self.test_set, batch_size=8)
        self.model = PTuneForLAMA(args, self.device, self.args.template)

        if self.args.load_checkpoint_dir is not None:
            # path = join(self.args.load_checkpoint_dir, self.args.model_name, self.args.relation_id)
            path = join(self.args.load_checkpoint_dir, self.args.relation_id)
            file_name = os.listdir(path)[0]
            path = join(path, file_name)
            print('-'*5)
            print('Load pretrained prompt checkpoint from:', path, '...')
            checkpoint = torch.load(path)
            if checkpoint['saved_params_type'] == 'fine-tuning':
                self.model.load_state_dict(checkpoint['embedding'], strict=False)
            elif checkpoint['saved_params_type'] == 'adapter':
                self.model.load_state_dict(checkpoint['embedding'], strict=False)
            else:
                self.model.prompt_encoder.load_state_dict(checkpoint['embedding'])
            print('Loading finished')

    def get_TREx_parameters(self):
        relation = load_file(join(
            self.args.data_dir, "single_relations/{}.jsonl".format(self.args.relation_id)))[0]
        data_path_pre = "fact-retrieval/original/{}/".format(
            self.args.relation_id)
        data_path_post = ".jsonl"
        return relation, data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
            model = self.best_model
        else:
            loader = self.dev_loader
            dataset = self.dev_set
            model = self.model
        with torch.no_grad():
            model.eval()
            kld, hit1, loss, mrr, counter_entropy, counter_hit1 = 0, 0, 0, 0, 0, 0
            top1_pred = None
            top1_truth = None
            for x_hs, x_ts in loader:
                if evaluate_type == 'Test':
                    _loss, _kld, _hit1, top10, _mrr, _counter_entropy, _counter_hit1 = model(
                        x_hs, x_ts, mask_mode=self.args.data_mask, return_candidates=True)
                    if top1_pred is None:
                        top1_truth = top10[-1] # (bz)
                        top1_pred = torch.tensor([i[0] for i in top10[0]])
                    else:
                        top1_pred  = torch.cat([top1_pred, torch.tensor([i[0] for i in top10[0]])], dim=0)
                        top1_truth = torch.cat([top1_truth, top10[-1]], dim=0)
                else:
                    _loss, _kld, _hit1, _mrr, _counter_entropy, _counter_hit1 = model(x_hs, x_ts, mask_mode=self.args.data_mask)
                
                kld += _kld
                hit1 += _hit1
                mrr += _mrr
                counter_entropy += _counter_entropy
                counter_hit1 += _counter_hit1
                loss += _loss.item()

            kld /= len(dataset)
            hit1 /= len(dataset)
            counter_hit1 /= len(dataset)
            mrr /= len(dataset)
            counter_entropy /= len(dataset)

            if evaluate_type == 'Test':
                predictions_np = top1_pred.cpu().numpy()
                targets_np = top1_truth.squeeze(-1).cpu().numpy()
                macro_f1 = f1_score(targets_np, predictions_np, average='macro')
                print("{} Epoch {} Loss: {:.4f}".format(self.args.relation_id, epoch_idx,
                                                          loss / len(dataset)))
                print("{} kld: {:.4f} Hit@1: {:.4f}, F1: {:4f}, MRR: {:.4f}, ct_ent: {:.4f}, counter_hit1: {:.4f}".format(evaluate_type, kld, hit1, macro_f1, mrr, counter_entropy, counter_hit1))
                return loss, kld, hit1, macro_f1, mrr, counter_entropy, counter_hit1

            print("{} Epoch {} Loss: {:.4f}".format(self.args.relation_id, epoch_idx,
                                                          loss / len(dataset)))
            print("{} kld: {:.4f} Hit@1: {:.4f}, MRR: {:.4f}, ct_ent: {:.4f}, counter_hit1: {:.4f}".format(evaluate_type, kld, hit1, mrr, counter_entropy, counter_hit1))
        return loss, kld, hit1, mrr, counter_entropy, counter_hit1
    

    def evaluate_consist(self):
        loader = self.test_loader
        dataset = self.test_set
        id_all_top1_preds = None
        pr_all_top1_preds = None
        label_all_ids = None
        with torch.no_grad():
            self.model.eval()
            id_stat = torch.tensor([0.0]*4, device=self.device) #id_consistency, id_all_consistency, id_acc_consistency, id_hit
            pr_stat = torch.tensor([0.0]*4, device=self.device) #pr_consistency, pr_all_consistency, pr_acc_consistency, pr_hit
            for x_hs, x_ts in loader:
                id_top1_preds, pr_top1_preds, label_ids = self.model.cal_consistency(
                        x_hs, x_ts)
                if id_all_top1_preds is None:
                    id_all_top1_preds = id_top1_preds
                    pr_all_top1_preds = pr_top1_preds
                    label_all_ids = label_ids
                else:
                    id_all_top1_preds = torch.cat((id_all_top1_preds, id_top1_preds), dim=0) # n_sample, n_rel
                    pr_all_top1_preds = torch.cat((pr_all_top1_preds, pr_top1_preds), dim=0) # n_sample, n_rel
                    label_all_ids = torch.cat((label_all_ids, label_ids), dim=0) # n_sample, 1
                id_out = cal_consistency(id_top1_preds, label_ids) # consist, all_consist, acc_consist, avg_p_hit_per_x
                pr_out = cal_consistency(pr_top1_preds, label_ids)
                for i, s in enumerate(id_out):
                    id_stat[i] += torch.sum(s)
                for i, s in enumerate(pr_out):
                    pr_stat[i] += torch.sum(s)
            id_stat /= len(dataset)
            pr_stat /= len(dataset)

            targets_np = label_all_ids.squeeze().cpu().numpy()
            id_macro_f1 = np.array([])
            pr_macro_f1 = np.array([])
            for i in range(id_all_top1_preds.size(1)):
                id_predictions_np = id_all_top1_preds[:,i].cpu().numpy()
                id_macro_f1 = np.append(id_macro_f1, f1_score(targets_np, id_predictions_np, average='macro'))
            for i in range(pr_all_top1_preds.size(1)):
                pr_predictions_np = pr_all_top1_preds[:,i].cpu().numpy()
                pr_macro_f1 = np.append(pr_macro_f1, f1_score(targets_np, pr_predictions_np, average='macro'))
            id_macro_f1 = np.mean(id_macro_f1)
            pr_macro_f1 = np.mean(pr_macro_f1)
        
        return id_stat, id_macro_f1, pr_stat, pr_macro_f1

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(
                     self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, str(self.args.seed), self.args.model_name, self.args.run_name,
                    self.args.relation_id)

    def get_adapter_params(self):
        adapter_params = {}
        for name, param in self.best_model.named_parameters():
            if "adapter" in name: 
                adapter_params[name] = param.data
        return adapter_params

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        if self.args.use_lm_finetune is True:
            embedding = self.best_model.state_dict()
            mode = 'fine-tuning'
        elif self.args.use_adaptor is True:
            embedding = self.get_adapter_params()
            mode = 'adapter'
        else:
            # prompt_tuning
            embedding = self.best_model.prompt_encoder.state_dict()
            mode = 'prompt'
        return {'embedding': embedding,
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args,
                'saved_params_type': mode}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        try:
            print("# Prompt:", self.model.prompt_encoder)
        except:
            print('No')
        print("# {} Checkpoint {} saved.".format(
            self.args.relation_id, ckpt_name))

    def train(self):
        if not self.args.only_evaluate:
            best_dev, dev_test, early_stop, has_adjusted = 0, 0, 0, True
            best_ckpt = None
            params = [{'params': self.model.prompt_encoder.parameters()}]
            if self.args.use_adaptor or self.args.use_lm_finetune:
                params.append(
                    {'params': self.model.model.parameters(), 'lr': self.args.ad_lr})
            optimizer = torch.optim.Adam(
                params, lr=self.args.pt_lr, weight_decay=self.args.weight_decay)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=self.args.decay_rate)
        
        for epoch_idx in range(100):
            # check early stopping
            self.model.entropy_mask_out = None
            if epoch_idx > -1:
                dev_loss, dev_kld, dev_hit1, dev_MRR, dev_ent, dev_counter_hit1 = self.evaluate(epoch_idx, 'Dev')
                if epoch_idx == 0:
                    self.best_model = self.model
                    test_loss, test_kld, test_hit1, test_f1, test_MRR, test_ent, test_ct_hit1 = self.evaluate(epoch_idx, 'Test')
                    self.best_ckpt = self.get_checkpoint(
                        epoch_idx, dev_hit1, test_hit1)
                    best_dev = dev_hit1
                    dev_test_hit1 = test_hit1
                    dev_test_f1 = test_f1
                    dev_test_MRR = test_MRR
                    dev_test_ent = test_ent
                    dev_test_kld = test_kld
                    dev_test_ct_hit1 = test_ct_hit1
                elif epoch_idx > 0 and (dev_hit1 >= best_dev) or self.args.only_evaluate:
                    self.best_model = self.model
                    early_stop = 0
                    best_dev = dev_hit1
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        test_loss, test_kld, test_hit1, test_f1, test_MRR, test_ent, test_ct_hit1 = self.evaluate(epoch_idx, 'Test')
                        self.best_ckpt = self.get_checkpoint(
                            epoch_idx, dev_hit1, test_hit1)
                        early_stop = 0
                        best_dev = dev_hit1
                        dev_test_hit1 = test_hit1
                        dev_test_f1 = test_f1
                        dev_test_MRR = test_MRR
                        dev_test_ent = test_ent
                        dev_test_kld = test_kld
                        dev_test_ct_hit1 = test_ct_hit1
                        print('-'*20)
                        print(f'dev_hit1:{best_dev}, test_hit1:{dev_test_hit1}, test_f1:{dev_test_f1}, test_MRR:{dev_test_MRR}')
                        print(f'test_ent:{dev_test_ent}, test_ct_hit1:{dev_test_ct_hit1}, test_kld:{dev_test_kld}')
                        # cal consistency
                        if self.args.relation_id not in self.omitted_relation:
                            # Relations that are hard to create patterns for
                            # or their subjects are mixed
                            id_stat, id_macro_f1, pr_stat, pr_macro_f1 = self.evaluate_consist()
                        else:
                            id_stat, id_macro_f1, pr_stat, pr_macro_f1 = self.evaluate_consist()
                            id_stat, pr_stat = [[0]*4]*2
                        print(f'id_consist:{id_stat[0]}, id_all_consist:{id_stat[1]}, id_acc_consist:{id_stat[2]}, id_hit:{id_stat[3]}, id_f1:{id_macro_f1}')
                        print(f'pr_consist:{pr_stat[0]}, pr_all_consist:{pr_stat[1]}, pr_acc_consist:{pr_stat[2]}, pr_hit:{pr_stat[3]}, pr_f1:{pr_macro_f1}')
                        print("{} Early stopping at epoch {}.".format(
                            self.args.relation_id, epoch_idx))
                        self.save(self.best_ckpt)

                        return (best_dev, dev_test_hit1, dev_test_f1, dev_test_MRR, dev_test_ent, dev_test_ct_hit1, dev_test_kld), id_stat, id_macro_f1, pr_stat, pr_macro_f1
            if self.args.only_evaluate:
                break

            # run training
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                loss, batch_hit1 = self.model(batch[0], batch[1])
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
        
        test_loss, test_kld, test_hit1, test_f1, test_MRR, test_ent, test_ct_hit1 = self.evaluate(epoch_idx, 'Test')
        self.best_ckpt = self.get_checkpoint(
            epoch_idx, dev_hit1, test_hit1)
        early_stop = 0
        best_dev = dev_hit1
        dev_test_hit1 = test_hit1
        dev_test_f1 = test_f1
        dev_test_MRR = test_MRR
        dev_test_ent = test_ent
        dev_test_kld = test_kld
        dev_test_ct_hit1 = test_ct_hit1
        print('-'*20)
        print(f'dev_hit1:{best_dev}, test_hit1:{dev_test_hit1}, test_f1:{dev_test_f1}, test_MRR:{dev_test_MRR}')
        print(f'test_ent:{dev_test_ent}, test_ct_hit1:{dev_test_ct_hit1}, test_kld:{dev_test_kld}')
        # cal consistency

        if self.args.relation_id not in self.omitted_relation:
            # Relations that are hard to create patterns for
            # or their subjects are mixed
            id_stat, id_macro_f1, pr_stat, pr_macro_f1 = self.evaluate_consist()
        else:
            id_stat, id_macro_f1, pr_stat, pr_macro_f1 = self.evaluate_consist()
            id_stat, pr_stat = [[0]*4]*2
        print(f'id_consist:{id_stat[0]}, id_all_consist:{id_stat[1]}, id_acc_consist:{id_stat[2]}, id_hit:{id_stat[3]}, id_f1:{id_macro_f1}')
        print(f'pr_consist:{pr_stat[0]}, pr_all_consist:{pr_stat[1]}, pr_acc_consist:{pr_stat[2]}, pr_hit:{pr_stat[3]}, pr_f1:{pr_macro_f1}')

        print("{} Normal stopping at epoch {}.".format(
            self.args.relation_id, epoch_idx))

        self.save(self.best_ckpt)

        return (best_dev, dev_test_hit1, dev_test_f1, dev_test_MRR, dev_test_ent, dev_test_ct_hit1, dev_test_kld), id_stat, id_macro_f1, pr_stat, pr_macro_f1


def main():
    print('gpu:', torch.cuda.is_available())
    all_stat_name = [
    'dev_hit',
    'test_hit', 
    'test_f1',
    'MRR',
    'counter_entropy',
    'ct_hit1',
    'kld',
    'id_consist',
    'id_all_consist',
    'id_acc_consist',
    'id_hit1',
    'id_macro',
    'pr_consist',
    'pr_all_consist',
    'pr_acc_consist',
    'pr_hit1',
    'pr_macro',
    ]
    all_stats={}
    for stat_name in all_stat_name:
        all_stats[stat_name] = []
    log_path = None
    args = construct_generation_args()

    relation_ids = os.listdir('../data/fact-retrieval/original')
    omitted_relation_ids = ['P527', 'P31', 'P166', 'P69', 'P47', 'P463', 'P101', 'P1923', 'P106', 'P530', 'P27', 'P178', 'P1412', 'P108', 'P39', 'P937', 'P1303', 'P190', 'P1001'] # only omit for calculating consistency
    print('All relations:', relation_ids)
    remained_rel = [r for r in relation_ids if r not in omitted_relation_ids]
    print('Num of relations used for computing consistency:', len(remained_rel))

    for relation_id in tqdm(relation_ids):
        if relation_id in ['P31', 'P527']:
            # bad relation for ParaRels
            continue
        args = construct_generation_args()
        if relation_id:
            args.relation_id = relation_id
        if type(args.template) is not tuple:
            args.template = eval(args.template)
        assert type(args.template) is tuple
        trainer = Trainer(args)
        if not log_path:
            log_path = os.path.join(trainer.get_save_path(), '../log.csv')
            with open(log_path, 'w') as fout:
                fout.write('relation, dev_acc, test_acc, test_f1, MRR, ct_entropy, ct_hit1, kld\n')
            log_path_cst = os.path.join(trainer.get_save_path(), '../log_consist.csv')
            with open(log_path_cst, 'w') as fout:
                fout.write('relation, ID_consist, ID_all_consist, ID_acc_consist, ID_hit1, ID_f1, PR_consist, PR_all_consist, PR_acc_consist, PR_hit1, PR_f1\n')

        main_stat, id_stat, id_marco, pr_stat, pr_marco = trainer.train()

        for stat_idx, stat_name in enumerate(['dev_hit', 'test_hit', 'test_f1', 'MRR', 'counter_entropy', 'ct_hit1', 'kld']):
            all_stats[stat_name].append(main_stat[stat_idx])
        for stat_idx, stat_name in enumerate(['id_consist', 'id_all_consist', 'id_acc_consist', 'id_hit1']):
            all_stats[stat_name].append(id_stat[stat_idx])
        all_stats['id_macro'].append(id_marco)
        for stat_idx, stat_name in enumerate(['pr_consist', 'pr_all_consist', 'pr_acc_consist', 'pr_hit1']):
            all_stats[stat_name].append(pr_stat[stat_idx]) 
        all_stats['pr_macro'].append(pr_marco)   
            
        with open(log_path, 'a') as fout:
            fout.write('{}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}\n'.format(
                relation_id, main_stat[0], main_stat[1], main_stat[2], main_stat[3], main_stat[4], main_stat[5], main_stat[6]))
        with open(log_path_cst, 'a') as fout:
            fout.write('{}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}\n'.format(
                relation_id, id_stat[0], id_stat[1], id_stat[2], id_stat[3], id_marco, pr_stat[0], pr_stat[1], pr_stat[2], pr_stat[3], pr_marco))
    
    stat_output = []
    cst_output = []
    for name, stat in all_stats.items():
        if ('pr' in name) or ('id' in name):
            if 'macro' in name:
                cst_output.append(sum(stat)/len(stat))
            else:
                cst_output.append(sum(stat)/len(remained_rel))
        else:
            stat_output.append(sum(stat)/len(stat))

    with open(log_path, 'a') as fout:
        fout.write('{}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}\n'.format('Average', stat_output[0], stat_output[1], stat_output[2], stat_output[3], stat_output[4], stat_output[5], stat_output[6]))
    with open(log_path_cst, 'a') as fout:
        fout.write('{}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}\n'.format('Average', cst_output[0], cst_output[1], cst_output[2], cst_output[3], cst_output[4], cst_output[5], cst_output[6], cst_output[7], cst_output[8], cst_output[9]))


if __name__ == '__main__':
    main()

