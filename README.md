# Rosetta 🌹
An NLP pipeline for job description extraction, skill clustering, and skill gap analysis

## Roadmap

### Phase 1: NER model
- [x] Load and preprocess SkillSpan dataset (BIO tagging, subword alignment)
- [x] Fine-tune distilbert-base-uncased on token classification (B-Skill, I-Skill, B-Knowledge, I-Knowledge, O)
- [ ] Evaluate on SkillSpan test split (precision, recall, F1 via seqeval)
- [ ] Compare against LLM extraction baseline (Parsec + Claude/GPT)
- [ ] Train on gold + silver (LLM-generated) data, ablation study across data strategies

### Phase 2: Taxonomy construction
- [ ] Run NER across bulk job posting corpus (10k+ JDs)
- [ ] Embed extracted skill spans with sentence-transformers
- [ ] Cluster with HDBSCAN to discover canonical skill nodes
- [ ] Build confidence-based routing: known skill / novel skill / review queue
- [ ] Bootstrap hierarchical taxonomy using ESCO ontology

### Phase 3: Serving and API
- [ ] FastAPI endpoint for real-time skill extraction from raw text
- [ ] Taxonomy lookup and normalization layer
- [ ] Feedback endpoint for flagging incorrect normalizations

### Phase 4: Applications
- [ ] JD comparison tool -- extract and diff skills between two job postings
- [ ] Skill gap analyzer -- compare resume skills against a target JD
- [ ] Trend tracker -- plot skill demand over time across a timestamped corpus
- [ ] Live demo frontend with highlighted entity extraction


## Citation

If you use this code or the SkillSpan dataset, please cite:

```bibtex
@inproceedings{zhang-etal-2022-skillspan,
    title = "{S}kill{S}pan: Hard and Soft Skill Extraction from {E}nglish Job Postings",
    author = "Zhang, Mike  and
      Jensen, Kristian N{\o}rgaard  and
      Sonniks, Sif  and
      Plank, Barbara",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    url = "https://aclanthology.org/2022.naacl-main.366",
    pages = "4962--4984",
}
