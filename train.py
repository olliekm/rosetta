import json
import numpy as np
from seqeval import metrics as seqeval_metrics
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer, 
                          DataCollatorForTokenClassification)

label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Dataset: SkillSpan (Zhang et al., 2022)
# https://aclanthology.org/2022.naacl-main.366

# load the data from skillspan
def load_skillspan(file_path):
    example = []
    with open(file_path) as f:
        for line in f:
            row = json.loads(line)
            example.append({
                "tokens": row['tokens'],
                "tags": merge_tags(row['tags_skill'], row['tags_knowledge'])
            })

    return example

# merging tags to avoid loss of data
def merge_tags(skill_tags, knowledge_tags):
    merged = []
    for s, k in zip(skill_tags, knowledge_tags):
        if s == "B":
            merged.append("B-Skill")
        elif s == "I":
            merged.append("I-Skill")
        elif k == "B":
            merged.append("B-Knowledge")
        elif k == "I":
            merged.append("I-Knowledge")
        else:
            merged.append("O")
    return merged

# load in data
train_data = load_skillspan('./json/train.json')
test_data = load_skillspan('./json/test.json')
dev_data = load_skillspan('./json/dev.json')

# quick sanity check
# for ex in train_data[:200]:
#     if any(t != "O" for t in ex["tags"]):
#         for token, tag in zip(ex["tokens"], ex["tags"]):
#             print(f"{token:20s} {tag}")
#         print("---")
#         break

def tokenize_and_align(example, tokenizer, label2id):
    tokenized = tokenizer(
        example['tokens'],
        is_split_into_words=True,
        truncation=True,
        padding=False
    )

    labels = []
    previous_word_idx = None
    for word_idx in tokenized.word_ids():
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["tags"][word_idx]])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    
    tokenized["labels"] = labels
    return tokenized

label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

train_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in train_data]
dev_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in dev_data]
test_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in test_data]

train_dataset = Dataset.from_list(train_tokenized)
dev_dataset = Dataset.from_list(dev_tokenized)
test_dataset = Dataset.from_list(test_tokenized)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir = './skillgraph-ner',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds = []
        seq_labels = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue

            seq_preds.append(id2label[int(p)])
            seq_labels.append(id2label[int(l)])

        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": seqeval_metrics.precision_score(true_labels, true_preds),
        "recall": seqeval_metrics.recall_score(true_labels, true_preds),
        "f1": seqeval_metrics.f1_score(true_labels, true_preds),
    }

model = AutoModelForTokenClassification.from_pretrained("./skillgraph-ner/checkpoint-3000")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print(results)

# trainer.train()