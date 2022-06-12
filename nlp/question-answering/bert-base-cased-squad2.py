from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

if __name__ == '__main__':
    model_name = "deepset/bert-base-cased-squad2"

    pipeline_model = pipeline("question-answering", model=model_name, tokenizer=model_name)
    question = "When did BBC Japan start broadcasting?"
    context = "BBC Japan was a general entertainment Channel.\n" + \
              "Which operated between December 2004 and April 2006.\n" + \
              "It ceased operations after its Japanese distributor folded."
    QA_input = {'question': question, 'context': context}

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    res = pipeline_model(QA_input)
    print(res)
