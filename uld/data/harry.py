import os
import random
from collections import defaultdict
import datasets
from datasets import load_dataset, Dataset

from .datamodule import TrainDataModule
from .conv_util import create_template
from ..harryutil.convert_data import get_HPQA, get_WikiText2, get_C4


class HarryPotterDataModule(TrainDataModule):
    def __init__(self, split, tokenizer, conv_template_config, max_len=1000, batch_size=4, with_retain=False, expand_forget=False, with_perturb=False, with_dpo=False, **kwargs) -> None:
        tokenizer.padding_side = 'left'
        tokenizer.trunation_side = 'left'
        print("Tokenizer set to left")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.dpo_mode = with_dpo
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        #! prepare train data
        self.forget_data = get_HPQA("hp_train_qa_200", num=400)
        self.forget_length = len(self.forget_data)

        if expand_forget:
            pass
        
        self.retain_length = 0 
        if with_retain:
            tmp_retain_data = get_C4(tokenizer, num=400)
            self.forget_data = datasets.concatenate_datasets([self.forget_data, tmp_retain_data])
            self.retain_length = len(tmp_retain_data)
        
        if with_perturb:
            pass
        
        #! prepare eval data
        self.wiki_eval = get_WikiText2(tokenizer, split='test', num=400, seed=42)
        self.forget_eval = get_HPQA("hp_train_qa_200", num=400)

        self.eval_sets = {
            'forget': self.forget_eval,
            'wiki': self.wiki_eval,
        }