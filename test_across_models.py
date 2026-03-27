import json
import time
import numpy as np
import pandas as pd
import torch
from collections import Counter
from seqeval import metrics as seqeval_metrics
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer, 
                          DataCollatorForTokenClassification)

label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]

test_models = [
    {"name": "distilbert-base-uncased"},
    {"name": "distilbert-base-cased"},
    {"name": "bert-base-uncased"},
    {"name": "bert-base-cased"},
    {"name": "jjzha/jobbert-base-cased"},
    {"name": "microsoft/deberta-v3-base", "add_prefix_space": True},
    {"name": "roberta-base", "add_prefix_space": True},
    {"name": "distilroberta-base", "add_prefix_space": True},
]

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
    report = seqeval_metrics.classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": seqeval_metrics.precision_score(true_labels, true_preds),
        "recall": seqeval_metrics.recall_score(true_labels, true_preds),
        "f1": seqeval_metrics.f1_score(true_labels, true_preds),
        "knowledge_f1": report.get("Knowledge", {}).get("f1-score", 0),
        "skill_f1": report.get("Skill", {}).get("f1-score", 0),
        "skill_precision": report.get("Skill", {}).get("precision", 0),
        "skill_recall": report.get("Skill", {}).get("recall", 0),
        "knowledge_precision": report.get("Knowledge", {}).get("precision", 0),
        "knowledge_recall": report.get("Knowledge", {}).get("recall", 0),
    }

label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# load in data
train_data = load_skillspan('./json/train.json')
test_data = load_skillspan('./json/test.json')
dev_data = load_skillspan('./json/dev.json')

all_results = []

for model_config in test_models:
    model_name = model_config["name"]
    add_prefix_space = model_config.get("add_prefix_space", False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=add_prefix_space)

    # Dataset: SkillSpan (Zhang et al., 2022)
    # https://aclanthology.org/2022.naacl-main.366

    # load the data from skillspan


    # quick sanity check
    # for ex in train_data[:200]:
    #     if any(t != "O" for t in ex["tags"]):
    #         for token, tag in zip(ex["tokens"], ex["tags"]):
    #             print(f"{token:20s} {tag}")
    #         print("---")
    #         break


    train_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in train_data]
    dev_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in dev_data]
    test_tokenized = [tokenize_and_align(ex, tokenizer, label2id) for ex in test_data]

    train_dataset = Dataset.from_list(train_tokenized)
    dev_dataset = Dataset.from_list(dev_tokenized)
    test_dataset = Dataset.from_list(test_tokenized)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./model-comparison/{model_name.replace("/", "_")}',
        num_train_epochs=5,
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

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,  # e.g., "bert-base-cased"
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # training!
    start = time.time()
    trainer.train()
    train_time = time.time() - start

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    confusion = Counter()
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            pred_label = id2label[p]
            true_label = id2label[l]
            if pred_label != true_label:
                confusion[(true_label, pred_label)] += 1

    # saving error samples
    samples = []
    for i in range(min(20, len(test_data))):
        tokens = test_data[i]["tokens"]
        true_tags = test_data[i]["tags"]
        pred_tags = [id2label[p] for p, l in zip(preds[i], labels[i]) if l != -100]
        samples.append({"tokens": tokens, "true": true_tags, "pred": pred_tags})

    with open(f"error_samples_{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(samples, f, indent=2)


    # get inference lantency
    model.eval()
    batch_samples = [test_dataset[i] for i in range(min(32, len(test_dataset)))]
    batch = data_collator(batch_samples)
    batch = {k: v.to(model.device) for k, v in batch.items()}


    total_inference_time = 0
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            model(**batch)
            
            inference_time = time.time() - start
            total_inference_time += inference_time
    avg_inference_time = total_inference_time / (100 * len(batch_samples))

    # check overfitting
    dev_results = trainer.evaluate(dev_dataset)
    test_results = trainer.evaluate(test_dataset)

    test_results["model"] = model_name
    test_results["train_time"] = train_time
    test_results["num_params"] = model.num_parameters()
    test_results["avg_inference_latency"] = avg_inference_time
    test_results["dev_f1"] = dev_results["eval_f1"]
    test_results["overfit_gap"] = dev_results["eval_f1"] - test_results["eval_f1"]
    test_results["top_confusion"] = confusion.most_common(5)

    all_results.append(test_results)

df = pd.DataFrame(all_results)
print(df.sort_values("f1", ascending=False))
df.to_csv("model_comparison.csv", index=False)