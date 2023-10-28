try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pandas as pd
from razdel import tokenize
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer 
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import logging
from transformers.trainer import logger as noisy_logger
noisy_logger.setLevel(logging.WARNING)
import numpy as np
from tqdm import tqdm

def tokenize_(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    return tokenized_inputs

def extract_tokens(item):
    
    # воспользуемся удобным токенайзером из библиотеки razdel, 
    # она помимо разбиения на слова, сохраняет важные для нас числа - начало и конец слова в токенах
    
    raw_toks = list(tokenize(item['video_info']))
    words = [tok.text for tok in raw_toks]
    # присвоим для начала каждому слову тег 'О' - тег, означающий отсутствие NER-а
    char2word = [None] * len(item['video_info'])
    # так как NER можем состаять из нескольких слов, то нам нужно сохранить эту инфорцию
    for i, word in enumerate(raw_toks):
        char2word[word.start:word.stop] = [i] * len(word.text)    
    return {'tokens': words}

label_list = ['O', 'B-Дата', 'B-бренд', 'B-вид спорта', 'B-видеоигра', 'B-команда', 'B-лига', 'B-локация', 'B-модель', 'B-название проекта', 'B-организация', 'B-персона', 'B-сезон', 'B-серия', 'I-Дата', 'I-бренд', 'I-вид спорта', 'I-видеоигра', 'I-команда', 'I-лига', 'I-локация', 'I-модель', 'I-название проекта', 'I-организация', 'I-персона', 'I-сезон', 'I-серия']
data = pd.read_csv("train_data/ner_data_train.csv")

ner_data = [extract_tokens(item) for i, item in data.iterrows()]


ner_dataset = DatasetDict({
    'test': Dataset.from_pandas(pd.DataFrame(ner_data))
})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred = np.zeros((len(ner_data), 512, len(label_list)))

ALL_RESULTS = []
for i, model_checkpoint in enumerate(['Babelscape', 'Davlan', 'DunnBC22', 'Tirendaz']):
#for i, model_checkpoint in enumerate(['Babelscape']):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device=device)    
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    model.config.id2label = dict(enumerate(label_list))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    tokenized_datasets = ner_dataset.map(tokenize_, batched=True)
    trainer = Trainer(
        model,
        tokenizer=tokenizer,
    )
    
    predictions, _, _ = trainer.predict(tokenized_datasets["test"])
    predictions = torch.softmax(torch.from_numpy(predictions), axis = -1).numpy()
    print(predictions.shape)

    model_predictions_probas = []
    ##predictions = np.argmax(predictions, axis=2)
    for i, elem in tqdm(enumerate(ner_data), total = len(ner_data)):
        tokens = tokenizer(elem['tokens'], truncation=True, is_split_into_words=True, return_tensors='pt')
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        token_text = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

        labels = []
        probas = predictions[i, :]
        for t, proba in zip(token_text, probas):
            if '##' not in t:
                #labels.append(label_list[idx])
                labels.append(proba)
        model_predictions_probas.append(np.array(labels))
    ALL_RESULTS.append(model_predictions_probas)

submission_ = pd.DataFrame(columns=[['video_info', 'entities_prediction']])
submission_['entities_prediction'] = submission_['entities_prediction'].astype('object')
for i in range(len(ALL_RESULTS[0])):
    probas1 = ALL_RESULTS[0][i]
    probas2 = ALL_RESULTS[1][i]
    probas3 = ALL_RESULTS[2][i]
    probas4 = ALL_RESULTS[3][i]
    probas = probas1 + probas2 + probas3 + probas4
    indexes = np.argmax(probas, axis=-1)

    labels = []
    for ind in indexes:
        labels.append(label_list[ind])

    submission_.loc[i, 'video_info'] = data.iloc[i, 0]
    submission_.loc[i, 'entities_prediction'] = [[label] for label in labels]
submission_.to_csv('submission.csv', index=None)




















from transformers import pipeline

pipe = pipeline(model=model, tokenizer=tokenizer, task='ner', aggregation_strategy='average', device=device)

def predict_ner(text, tokenizer, model, pipe, verbose=True):
    tokens = tokenizer(text, truncation=True, is_split_into_words=True, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}    
    with torch.no_grad():
        pred = model(**tokens)
    # print(pred.logits.shape)
    indices = pred.logits.argmax(dim=-1)[0].cpu().numpy()
    token_text = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    print(len(indices), len(token_text))
    labels = []
    for t, idx in zip(token_text, indices):
        if '##' not in t:
            labels.append(label_list[idx])
        if verbose:    
            print(f'{t:15s} {label_list[idx]:10s}')
    return text, pipe(text), labels

from tqdm.notebook import tqdm

submission = pd.DataFrame(columns=[['video_info', 'entities_prediction']])
submission['entities_prediction'] = submission['entities_prediction'].astype('object')

for i, elem in tqdm(enumerate(ner_data), total = len(ner_data)):
    _, _, labels = predict_ner(elem['tokens'], tokenizer, model, pipe, verbose=False)
    submission.loc[i, 'video_info'] = elem
    submission.loc[i, 'entities_prediction'] = [[label] for label in labels]



# Remove ignored index (special tokens)
from datasets import load_metric
metric = load_metric("seqeval")
true_labels = [ner['tags'] for ner in ner_data]
pred_labels = [ner['entities_prediction'] for _, ner in submission_['entities_prediction'].iterrows()]
results = metric.compute(predictions=pred_labels, references=true_labels)

