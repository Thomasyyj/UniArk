import json
from os.path import join


def init_vocab(args):
    global shared_vocab, lama_vocab
    shared_vocab = json.load(open(join(args.data_dir, '29k-vocab.json')))
    lama_vocab = json.load(open(join(args.data_dir, '34k-vocab.json')))


def token_wrapper(args, token):
    if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
        return 'Ġ' + token
    else:
        return token


def get_vocab(model_name, strategy):
    if strategy == 'shared':
        if 'roberta' in model_name:
            return shared_vocab['roberta-large']
        else:
            assert model_name in shared_vocab
            return shared_vocab[model_name]
    elif strategy == 'lama':
        if 'roberta' in model_name:
            return lama_vocab['roberta-large']
        else:
            assert model_name in lama_vocab
            return lama_vocab[model_name]


def get_vocab_by_strategy(args, tokenizer):
    if args.vocab_strategy == 'original':
        return tokenizer.get_vocab()
    else:
        return get_vocab(args.model_name, args.vocab_strategy)

def check_vocab(tokenizer, token):
    tokenizer_name = tokenizer.__class__.__name__
    if 'Roberta' in tokenizer_name:
        if tokenizer.convert_tokens_to_ids(token) == 3:
            token_upper = token[0].upper() + token[1:]
            if tokenizer.convert_tokens_to_ids(token_upper) == 3:
                tokenUPPER = token.upper()
                if tokenizer.convert_tokens_to_ids(tokenUPPER) == 3:
                    # print(f'Unknown token found: {token}')
                    return None
                return tokenUPPER
            return token_upper
        return token
    elif 'Bert' in tokenizer_name:
        if tokenizer.convert_tokens_to_ids(token) == 100:
            token_upper = token[0].upper() + token[1:]
            if tokenizer.convert_tokens_to_ids(token_upper) == 100:
                tokenUPPER = token.upper()
                if tokenizer.convert_tokens_to_ids(tokenUPPER) == 100:
                    # print(f'Unknown token found: {token}')
                    return None
                return tokenUPPER
            return token_upper
        return token

