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
from copy import deepcopy

def extract_labels(item):
    
    # воспользуемся удобным токенайзером из библиотеки razdel, 
    # она помимо разбиения на слова, сохраняет важные для нас числа - начало и конец слова в токенах
    
    raw_toks = list(tokenize(item['video_info']))
    words = [tok.text for tok in raw_toks]
    # присвоим для начала каждому слову тег 'О' - тег, означающий отсутствие NER-а
    word_labels = ['O'] * len(raw_toks)
    char2word = [None] * len(item['video_info'])
    # так как NER можем состаять из нескольких слов, то нам нужно сохранить эту инфорцию
    for i, word in enumerate(raw_toks):
        char2word[word.start:word.stop] = [i] * len(word.text)

    labels = item['entities']
    if isinstance(labels, dict):
        labels = [labels]
    if labels is not None:
        for e in labels:
            if e['label'] != 'не найдено':
                e_words = sorted({idx for idx in char2word[e['offset']:e['offset']+e['length']] if idx is not None})
                if e_words:
                    word_labels[e_words[0]] = 'B-' + e['label']
                    for idx in e_words[1:]:
                        word_labels[idx] = 'I-' + e['label']
                else:
                    continue
            else:
                continue
        return {'tokens': words, 'tags': word_labels}
    else: return {'tokens': words, 'tags': word_labels}

    
def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids = [label_list.index(idx) if isinstance(idx, str) else idx for idx in label_ids]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
data = pd.read_csv("train_data/ner_data_train.csv")

df = data.copy()
df['entities'] = df['entities'].apply(lambda l: l.replace('\,', ',')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: l.replace('\\\\', '\\')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: '[' + l + ']'if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: json.loads(l)if isinstance(l, str) else l)

ner_data = [extract_labels(item) for i, item in df.iterrows()]

ner_train, ner_test = train_test_split(ner_data, test_size=0.1)

label_list = sorted({label for item in ner_train for label in item['tags']})
if 'O' in label_list:
    label_list.remove('O')
    label_list = ['O'] + label_list
# print(label_list)
ner_data = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(ner_train)),
    'test': Dataset.from_pandas(pd.DataFrame(ner_test))
})

#import telebot
#bot = telebot.TeleBot('')

model_checkpoints = ["Babelscape/wikineural-multilingual-ner",\
                     "Davlan/distilbert-base-multilingual-cased-ner-hrl",\
                    "DunnBC22/bert-base-multilingual-cased-fine_tuned-ner-WikiNeural_Multilingual",\
                    "Tirendaz/multilingual-xlm-roberta-for-ner"]

for model_checkpoint in model_checkpoints:
    output_directory = model_checkpoint.split('/')[0]
    
    #bot.send_message(312849799, 'new_start')
    batch_size = 8
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device=device, ignore_mismatched_sizes=True)
    
    tokenized_datasets = ner_data.map(tokenize_and_align_labels, batched=True)
    
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True)
    model.config.id2label = dict(enumerate(label_list))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    
    # Специальный объект для удобного формирования батчей
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    metric = load_metric("seqeval")
    
    # Для дообучения берта можно эксперементировать с заморозкой/разморозкой разных слоев, здесь мы оставим все слои размороженными
    # Для быстроты обучения можно заморозить всю бертовую часть, кроме классификатора, но тогда качесвто будет похуже
    num_layers = 0
    for param in model.parameters():
        num_layers += 1
    
    for param in model.parameters():
        param.requires_grad = True
    
    
            
    args = TrainingArguments(
        "ner",
        evaluation_strategy = "epoch",
        # logging_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to='none',
    
        )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # trainer.state.best_model_checkpoint
    trainer.evaluate()
    
    # Посчитаем метрики на отложенном датасете    
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = torch.softmax(torch.from_numpy(predictions), axis = -1).numpy()
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    cm = pd.DataFrame(
        confusion_matrix(sum(true_labels, []), sum(true_predictions, []), labels=label_list),
        index=label_list,
        columns=label_list
    )
    
    print(model_checkpoint)
    print(results['overall_f1'])
    for res in results:
        if type(results[res]) == dict:
            print(res, results[res]['f1'])
    
    
    # Сохранение токенизатора
    tokenizer.save_pretrained(output_directory)
    
    # Сохранение модели
    model.save_pretrained(output_directory)
    
    # Сохранение конфигурации модели (если необходимо)
    model.config.save_pretrained(output_directory)
    
    
    #bot.send_message(312849799, model_checkpoint)
    # del model, tokenizer, trainer, data_collator
    torch.cuda.empty_cache()




        
'''
Персона — челоке/персонаж, упоминаемый в тексте (Например: Настя Ивлеева, Дядя Федор, Баста, Евгений Онегин). Персона - только имя человека/персонажа. Например, фраза "Руководитель отдела" не является персоной.
Локация — географические координаты, такие как страна, город, край, область, название пространства (Например: Москва, Россия, Казахстан, на Фестивальной улице, гостица Солнечная)
Дата — дата события (Например: 4 января, в 1992 году, в октябре)
Организация — организация, орган государственной власти, учреждение (Например: Яндекс, Министерство цифрового развития, Совет Федерации, ООО "Рога и Копыта")
Бренд - бренд (Например:  Samsung, Audi, Toyota, )
Модель - название модели (Например: Galaxy S10, RS6, Corolla)
Название проекта - название шоу, сериала, фильма, проекта организации (Например: Битва экстрасенсов, беременна в 16, Интерны, Лидеры России, Международная кооперация и экспорт)
Сезон - сезон/часть сериала (Например: третий, первый, 1) 
Серия - серия сериала, передачи (Например: первая, заключительная, 4) 
Лига - название спортивной лиги (Например: Чемпионат Европы-2024, Лалига, Английская премьер лига, Восточно-европейская хоккейная лига, чемпионате мира)
Команда - команда (спортивная, студенческая) (Например: ЦСКА, Manchester United, КАМАЗ-мастер)
Вид спорта - название спорта или подтипа (Например: футбол, керлинг, автоспорт, танго)
Видеоигра — название видеоигры (Например: GTA, Call of Duty, Майнкрафт)
'''


'''
Babelscape/wikineural-multilingual-ner

0,52
Дата 0.6082877247849883
бренд 0.2709424083769634
вид спорта 0.3135391923990499
видеоигра 0.4491467576791809
команда 0.4276568501920614
лига 0.3189189189189189
локация 0.5909090909090909
модель 0.2707509881422925
название проекта 0.44141852461469405
организация 0.2708066581306018
персона 0.6658109470981938
сезон 0.4210526315789474
серия 0.42057488653555225

freeze_count = 50
0.5235534644037416
Дата 0.6254796623177282
бренд 0.2668298653610771
вид спорта 0.2830626450116009
видеоигра 0.49232821881254174
команда 0.4502369668246446
лига 0.3274853801169591
локация 0.5929726368159204
модель 0.3284823284823285
название проекта 0.44526901669758817
организация 0.2616760516727393
персона 0.6671662763466043
сезон 0.38202247191011235
серия 0.4642857142857143

Davlan/distilbert-base-multilingual-cased-ner-hrl
0.504
Дата 0.59185303514377
бренд 0.26462765957446804
вид спорта 0.20833333333333334
видеоигра 0.47346938775510206
команда 0.4423305588585018
лига 0.2784810126582279
локация 0.5641685429398521
модель 0.304
название проекта 0.4298919003603322
организация 0.2563927719058984
персона 0.6468235512975536
сезон 0.4125
серия 0.4250411861614498

DunnBC22/bert-base-multilingual-cased-fine_tuned-ner-WikiNeural_Multilingual
0,514
Дата 0.6123727486296004
бренд 0.29739776951672864
вид спорта 0.27510917030567683
видеоигра 0.4863349684653119
команда 0.44364508393285373
лига 0.2638522427440634
локация 0.5834130781499203
модель 0.3372319688109161
название проекта 0.44594180704441044
организация 0.2672329304148971
персона 0.6498002663115846
сезон 0.4175824175824176
серия 0.42703533026113666

Tirendaz/multilingual-xlm-roberta-for-ner
0.5404689281202821
Дата 0.6049856184084372
бренд 0.32770745428973275
вид спорта 0.40624999999999994
видеоигра 0.43005952380952384
команда 0.5037974683544303
лига 0.32258064516129037
локация 0.5984769816545518
модель 0.3183760683760684
название проекта 0.46159110350727117
организация 0.3110443275732532
персона 0.6930000000000001
сезон 0.42786069651741293
серия 0.4301470588235295
'''





























