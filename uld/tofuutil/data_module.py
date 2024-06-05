#! Adapted from ToFU repo
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from .utils import get_model_identifiers_from_yaml

class TextDatasetQA(Dataset):
    def __init__(
        self, 
        data_path,
        tokenizer, 
        conv_template,
        split=None, 
        question_key='question', 
        answer_key='answer',
        max_num=-1,
    ):
        super(TextDatasetQA, self).__init__()
        self.conv_template = conv_template
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset(data_path, split)["train"]
        if max_num != -1:
            self.data = self.data.select(range(min(len(self.data), max_num)))
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)
    
    def prepare_input_ids(self, 
        question, answer,  
        question_start_token, question_end_token, answer_token, 
        tokenizer=None, max_len=None
    ):
        tokenizer = tokenizer
        #! Important about the format
        new_question = question_start_token + " " + question + " " + question_end_token
        new_answer = answer_token + answer
        full_text = new_question + new_answer
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
        encoded = tokenizer(
            full_text, 
            add_special_tokens=True, 
            max_length=max_len, 
            truncation=True, 
        )
        pad_length = max_len - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        if len(encoded.input_ids) == max_len:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

        # change label to -100 for question tokens
        label = torch.tensor(label)
        label[:num_question_tokens] = -100
        return (
            torch.tensor(pad_input_ids),
            label,
            torch.tensor(pad_attention_mask),
        )    

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            tensor_data = self.prepare_input_ids(
                question, answer, tokenizer=self.tokenizer, max_len=self.conv_template.max_len,
                question_start_token=self.conv_template.question_start_token, question_end_token=self.conv_template.question_end_token, answer_token=self.conv_template.answer_token,
            )
            pad_input_ids_list.append(tensor_data[0])
            label_list.append(tensor_data[1])
            pad_attention_mask_list.append(tensor_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze()
        )
    

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss
