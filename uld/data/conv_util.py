import torch
from omegaconf import OmegaConf
from typing import List, Optional, Tuple, Any

def create_template(template_config : dict, tokenizer=None, max_len=200) -> Any:
    template_config = OmegaConf.create(template_config)
    template_config['max_len'] = max_len
    template = ConvTemplate(
        **template_config,
        tokenizer=tokenizer,
    )
    return template

class ConvTemplate:
    def __init__(self, question_start_token, question_end_token, answer_token, **kwargs) -> None:
        self.question_start_token = question_start_token
        self.question_end_token = question_end_token
        self.answer_token = answer_token
        self.max_len = kwargs.get('max_len', 200)
    
    def prepare_gen_prompt(self, question=None, answer=None, **kwargs):
        if question != None:
            return (
                self.question_start_token 
                + question.strip() 
                + self.question_end_token
                + self.answer_token
            ).strip()
        elif kwargs.get('prefix', None) is not None:
            prefix = kwargs.get('prefix', None)
            return (
                prefix.strip()
                + " "
            )
        elif kwargs.get('text', None) is not None:
            return ""
        else:
            print(kwargs.keys())
            raise ValueError("Unkown input for conv template")

    def prepare_prompt(self, question=None, answer=None, **kwargs):
        if question != None:
            return (
                self.question_start_token 
                + question.strip() 
                + self.question_end_token
                + self.answer_token
                + answer.strip()
            )
        elif kwargs.get('prefix', None) is not None:
            prefix = kwargs.get('prefix', None)
            continu = kwargs.get('continuation',None)
            return (
                prefix.strip()
                + " "
                + continu
            )
        elif kwargs.get('text', None) is not None:
            return kwargs.get('text')
        else:
            print(kwargs.keys())
            raise ValueError("Unkown input for conv template")
            
    def prepare_batch_prompt(self, question, choices, **kwargs):
        return [
            self.prepare_prompt(question=question, answer=choice) for choice in choices
        ]