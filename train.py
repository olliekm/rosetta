

from transformers import AutoModelForTokenClassification, AutoTokenizer
import json

label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list)
)


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

