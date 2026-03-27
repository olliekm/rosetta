import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm

# Label configuration
label_list = ["O", "B-Skill", "I-Skill", "B-Knowledge", "I-Knowledge"]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}  # Fixed: was {l: i for l, i in ...}

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(
    "./skillgraph-ner/checkpoint-3000",
    id2label=id2label,
    label2id=label2id
)

# Create NER pipeline with batching support
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=-1  # CPU; change to 0 for GPU
)

# Load dataset
ds = load_dataset("jacob-hugging-face/job-descriptions", split="train")


def chunk_text(text):
    """Split text into processable chunks."""
    chunks = []
    for line in text.split("\n"):
        for sent in line.split("."):
            sent = sent.strip()
            if len(sent) > 10:
                chunks.append(sent)
    return chunks


def extract_skills(text):
    """Extract skills and knowledge entities from text."""
    chunks = chunk_text(text)
    if not chunks:
        return []

    skills = []
    # Process in batches for better performance
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            results = ner_pipeline(batch)
            for entities in results:
                for ent in entities:
                    if ent["entity_group"] in ("Skill", "Knowledge"):
                        skills.append({
                            "text": ent["word"],
                            "type": ent["entity_group"],
                            "score": float(ent["score"])
                        })
        except Exception as e:
            print(f"Warning: Failed to process batch: {e}")
            continue
    return skills


def main():
    all_skills = []

    for row in tqdm(ds, desc="Extracting skills"):
        skills = extract_skills(row["job_description"])
        all_skills.extend(skills)

    print(f"Done. {len(all_skills)} total skill mentions extracted.")

    with open("raw_extractions.json", "w") as f:
        json.dump(all_skills, f, indent=2)

    print(f"Saved {len(all_skills)} extractions to raw_extractions.json")


if __name__ == "__main__":
    main()
