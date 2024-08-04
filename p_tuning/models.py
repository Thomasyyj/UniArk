from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

def create_model(args):
    MODEL_CLASS, _, MODEL_CONFIG = get_model_and_tokenizer_class(args)
    if 'roberta' in args.model_name:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    elif 'bert' in args.model_name:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    return model


def get_model_and_tokenizer_class(args):
    if 'bert' in args.model_name:
        return AutoModelForMaskedLM, AutoTokenizer, AutoConfig
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model_name:
        embeddings = model.bert.get_input_embeddings()
    else:
        raise NotImplementedError()
    return embeddings
