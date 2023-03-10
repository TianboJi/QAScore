import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
)
# from ast import literal_eval
import torch
from pathlib import Path
path = lambda p: Path(p).absolute().resolve()
import argparse
import string





class QAScore:
    def __init__(self, device='cpu', max_len=512):
        modelname = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(modelname)
        model = RobertaForMaskedLM.from_pretrained(modelname).to(device)
        mask_id = tokenizer.mask_token_id
        eos_id = tokenizer.eos_token_id 
        sep_id = tokenizer.sep_token_id
        # if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.sep_id = sep_id
        
    def remove_punc(self,text,lower=False):
        table = str.maketrans({e:" " for e in string.punctuation})
        new_text = text.translate(table).strip()
        words = new_text.split()
        if lower:
            return ' '.join(words).lower()
        else:
            return ' '.join(words)
        
    def text_to_id_list(self, text):
        text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return text_ids
    
    def id_list_to_text(self, text_ids):
        text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(text_ids))
        return text
    
    def id_list_to_model_input(self, text_id_list):
        m = self.max_len
        n = len(text_id_list)
        start = max(0,n-m)
        return torch.tensor(text_id_list[start:]).unsqueeze(0).to(self.device)
    
    def get_masked_answer(self, answer):
        mask_id = self.mask_id
        answer_ids = self.text_to_id_list(answer)
        masked_answer_list = []
        for i, _ in enumerate(answer_ids):
            masked_answer_ids = answer_ids[:i] + [mask_id] + answer_ids[i+1:]
            masked_answer_list.append(masked_answer_ids)
        return masked_answer_list
    
    def truncate_list(self, ori_list):
        n = len(ori_list)
        m = self.max_len
        start = max(0,n-m)
        return ori_list[start:]
    
    def get_input_and_label_by_pqa(self, paragraph, question, answer):

        eos_id = self.eos_id
        paragraph_ids = self.text_to_id_list(paragraph)
        question_ids = self.text_to_id_list(question)
        answer_ids = self.text_to_id_list(answer)
        
        label_ids = paragraph_ids + [eos_id] + question_ids + [eos_id] + answer_ids 
        label_ids = self.truncate_list(label_ids)
        
        masked_answer_id_lists = self.get_masked_answer(answer)
        
        pqa_id_list = [
            paragraph_ids + [eos_id] + question_ids + [eos_id] + masked_answer_ids
            for masked_answer_ids in masked_answer_id_lists
        ]
        pqa_id_list = [self.truncate_list(pqa_ids) for pqa_ids in pqa_id_list]
        return [(pqa_ids, label_ids) for pqa_ids in pqa_id_list]

    def get_single_score(self, paragraph, question, answer):
        input_and_label_list = self.get_input_and_label_by_pqa(paragraph, question, answer)
        scores = []
        with torch.no_grad():
            self.model.eval()
            for input_ids, label_ids in input_and_label_list:
                input_tensor = torch.tensor([input_ids]).to(self.device)
                label_tensor = torch.tensor([label_ids]).to(self.device)
                output = self.model(input_tensor,labels=label_tensor)
                score = output.loss.mean().item()
                scores.append(-score)
        return scores
    
    def score_by_file(self,paragraph_file, question_file, answer_file, use_tqdm=False):
        with path(paragraph_file).open() as f:
            paragraphs = f.read().splitlines()
        with path(question_file).open() as f:
            questions = f.read().splitlines()
        with path(answer_file).open() as f:
            answers = f.read().splitlines()
        result = self.get_model_score(paragraphs, questions, answers, use_tqdm=use_tqdm)
        return result
        
    def get_model_score(self, paragraphs, questions, answers, use_tqdm=False):
        overall = []
        if use_tqdm:
            iterator = tqdm(zip(paragraphs, questions, answers),total=len(paragraphs),desc="Evaluating")
        else:
            iterator = zip(paragraphs, questions, answers)
        
        for paragraph, question, answer in iterator:
            overall.append(self.get_single_score(paragraph, question, answer))
        return np.mean(overall)
    



def main():
     
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paragraph_file", type=str,
        help="path to paragraph file",
    )
    parser.add_argument(
        "--question_file", type=str,
        help="path to question file",
    )
    parser.add_argument(
        "--answer_file", type=str,
        help="path to answer file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="path to answer file",
    )
    parser.add_argument(
        "--remove_punctuation", action="store_true",
        help="whether to remove punctuations before evaluating",
    )
    parser.add_argument(
        "--show_progress", action="store_true",
        help="whether to show the progress bar when evaluating",
    )

    args, _ = parser.parse_known_args()
    qascorer = QAScore(device=args.device)
    result = qascorer.score_by_file(
        args.paragraph_file, 
        args.question_file, 
        args.answer_file,
        use_tqdm=args.show_progress
    )
    print(f"QAScore: {result}")

if __name__ == '__main__':
    main()