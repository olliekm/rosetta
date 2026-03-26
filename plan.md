
RAW job posting data: https://huggingface.co/datasets/xanderios/linkedin-job-postings
Run LLM extraction on the raw data for skills
Validate this vs the skillspan dataset
That gives you precision/recall numbers on your extraction prompt before you've trained anything. This is your LLM baseline.

https://github.com/kris927b/SkillSpan/

Use pretained NER model https://huggingface.co/docs/transformers/en/model_doc/distilbert
Fine-tune it on the LLM generated training data, evaluate it on the skill span dataset

Evaluate the difference in accuracy vs cost trade off

