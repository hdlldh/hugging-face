from transformers import BertModel, BertTokenizer, pipeline

if __name__ == '__main__':
    model_name = "bert-base-uncased"
    pipeline_model = pipeline("fill-mask", model=model_name, tokenizer=model_name)
    encoded_input = "The goal of life is [MASK]."

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    res = pipeline_model(encoded_input)
    print(res)
