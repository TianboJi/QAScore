# QAScore
Code for paper "QAScore - An Unsupervised Unreferenced Metric for the Question Generation Evaluation" (see this [link](https://www.mdpi.com/1099-4300/24/11/1514))

## Requirements
- python=3.8
- transformers
- pytorch
- tqdm

## Usage
run the following command
```
python -u QAScore.py \
        --paragraph_file example_files/paragraphs.txt \
        --question_file example_files/questions.txt \
        --answer_file example_files/answers.txt \
        --device "cuda" \
        --show_progress 
```
and the result is `-0.9693889460924203`
