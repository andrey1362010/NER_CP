try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pandas as pd
import json
from razdel import tokenize
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer 
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback, IntervalStrategy
import logging
from transformers.trainer import logger as noisy_logger
noisy_logger.setLevel(logging.WARNING)
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from transformers import DataCollatorForTokenClassification


import torch
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
# ner_train, ner_data = train_test_split(ner_data, test_size=0.2, random_state=1)

ner_data = DatasetDict({
    'test': Dataset.from_pandas(pd.DataFrame(ner_data))
})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i, model_checkpoint in enumerate(['Babelscape', 'Davlan', 'DunnBC22', 'Tirendaz']):    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device=device)    
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    # model.config.id2label = dict(enumerate(label_list))
    # model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    trainer = Trainer(
        model,
        tokenizer=tokenizer,
    )
    tokenized_datasets = ner_data.map(tokenize_, batched=True)
    predictions, _, _ = trainer.predict(tokenized_datasets["test"])
    predictions = torch.softmax(torch.from_numpy(predictions), axis = -1).numpy()
    if i == 0:
        pred = predictions
    else:
        pred += predictions

predictions = np.argmax(pred, axis=2)

submission_ = pd.DataFrame(columns=[['video_info', 'entities_prediction']])
submission_['entities_prediction'] = submission_['entities_prediction'].astype('object')
for i, elem in tqdm(enumerate(ner_data), total = len(ner_data)):
    tokens = tokenizer(elem['tokens'], truncation=True, is_split_into_words=True, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    token_text = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    labels = []
    indices = predictions[i, :]
    for t, idx in zip(token_text, indices):
        if '##' not in t:
            labels.append(label_list[idx])
    submission_.loc[i, 'video_info'] = data.iloc[i, 0]
    submission_.loc[i, 'entities_prediction'] = [[label] for label in labels]
submission_.to_csv('submission.csv', index = None)

