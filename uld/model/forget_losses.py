#! This file contains the implementation of multiple unlearn losses

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from transformers import AutoModelForCausalLM
from functools import partial

# Utility functions
def NextTokenPredictionLoss(model: AutoModelForCausalLM, input_ids, attention_mask, labels, position_ids=None):
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=labels, 
        position_ids=position_ids
    )
    assert outputs.loss is not None, "Forget loss is None"
    return outputs.loss

def TokenNextTokenPredictionLoss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss


def split_forget_retain(input_ids, attention_mask, labels=None, retainlabels=None):
    if retainlabels is not None:
        # Split the batch into forget/retain
        forget_input_ids = input_ids[retainlabels == 0]
        forget_attention_mask = attention_mask[retainlabels == 0]
        forget_labels = labels[retainlabels == 0]
 
        remember_input_ids = input_ids[retainlabels == 1] 
        remember_attention_mask = attention_mask[retainlabels == 1]
        remember_labels = labels[retainlabels == 1]

        return (
            (forget_input_ids, forget_attention_mask, forget_labels),
            (remember_input_ids, remember_attention_mask, remember_labels)
        )
    else:
        return (
            (input_ids, attention_mask, labels),
            (None, None, None)
        )

class ForgetRetainLoss:
    def __init__(self, forget_loss_func, retain_loss_func=None, retain_weight=1.0) -> None:
        self.forget_loss_func = forget_loss_func
        self.retain_loss_func = retain_loss_func
        self.retain_weight = retain_weight

    def calculate_loss(self, model, input_ids, attention_mask, labels=None, retainlabels=None, oracle_model=None, **kwargs):
        (
            (forget_input_ids, forget_attention_mask, forget_labels),
            (retain_input_ids, retain_attention_mask, retain_labels)
        ) = split_forget_retain(
            input_ids, attention_mask, labels, retainlabels=retainlabels
        )

        if forget_input_ids is None or forget_input_ids.shape[0] == 0:
            forget_loss = torch.tensor(0.).to(input_ids.device)
        else:
            forget_loss = self.forget_loss_func(model, input_ids=forget_input_ids, attention_mask=forget_attention_mask, labels=forget_labels, oracle_model=oracle_model)

        if retain_input_ids is None or retain_input_ids.shape[0] == 0:
            retain_loss = torch.tensor(0.).to(input_ids.device)
        else:
            retain_loss = self.retain_loss_func(model, input_ids=retain_input_ids, attention_mask=retain_attention_mask, labels=retain_labels, oracle_model=oracle_model)

        loss = forget_loss + self.retain_weight * retain_loss
        return loss, forget_loss, retain_loss

    def __call__(self, model, batch: Dict[str, Any], oracle_model=None) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'] # origin response
        
        if 'retainlabels' in batch:
            retainlabels = batch['retainlabels']
        else:
            retainlabels = None
        
        loss, forget_loss, retain_loss = self.calculate_loss(
            model, input_ids, attention_mask, labels, retainlabels, oracle_model=oracle_model
        )
        return {
            'loss': loss,
            'forget_loss': forget_loss,
            'retain_loss': retain_loss 
        }

# For RMU
class RMULoss(ForgetRetainLoss):
    def __init__(self, forget_loss_func, retain_loss_func, model_config, layerid, retain_weight=1200, steering_coeff=6.5) -> None:
        super().__init__(forget_loss_func, retain_loss_func, retain_weight=retain_weight)
        random_vector = torch.rand(1, 1, model_config.hidden_size, dtype=torch.float32)
        control_vec = random_vector / torch.norm(random_vector) * steering_coeff
        self.control_vec = control_vec
        self.layerid = layerid

        #! wrap new params for forget/retain loss
        self.forget_loss_func = partial(forget_loss_func, control_vec=control_vec, layerid=layerid)
        self.retain_loss_func = partial(retain_loss_func, control_vec=control_vec, layerid=layerid)
   

# For DPO
class PreferLoss(ForgetRetainLoss):
    def __init__(self, forget_loss_func, retain_loss_func=None, retain_weight=1.0) -> None:
        self.forget_loss_func = forget_loss_func
        self.retain_loss_func = retain_loss_func
        self.retain_weight = retain_weight

    def calculate_loss(self, model, input_ids, labels, label_attention_mask, prefer_input_ids, prefer_labels, prefer_label_attention_mask, retainlabels, oracle_model=None, **kwargs):
        # We only need preferlabel for 
        forget_input_ids = input_ids[retainlabels == 0]
        forget_attention_mask = label_attention_mask[retainlabels == 0]
        forget_labels = labels[retainlabels == 0]
        forget_prefer_input_ids = prefer_input_ids[retainlabels == 0]
        forget_prefer_attention_mask = prefer_label_attention_mask[retainlabels == 0]
        forget_prefer_labels = prefer_labels[retainlabels == 0]

        retain_input_ids = input_ids[retainlabels == 1]
        retain_labels = labels[retainlabels == 1]
        retain_attention_mask = label_attention_mask[retainlabels == 1]

        if forget_input_ids.shape[0] == 0:
            forget_loss = torch.tensor(0.).to(input_ids.device)
        else:
            forget_loss = self.forget_loss_func(
                model,
                forget_input_ids, 
                forget_labels, 
                forget_attention_mask, 
                forget_prefer_input_ids, 
                forget_prefer_labels, 
                forget_prefer_attention_mask,
                oracle_model=oracle_model
            )
        if retain_input_ids.shape[0] == 0:
            remember_loss = torch.tensor(0.).to(input_ids.device)
        else:
            remember_loss = self.retain_loss_func(
                model, input_ids=retain_input_ids, 
                labels=retain_labels, 
                attention_mask=retain_attention_mask, 
                oracle_model=oracle_model
            )

        loss = forget_loss + self.retain_weight * remember_loss
        return loss, forget_loss, remember_loss

    def __call__(self, model, batch: Dict[str, Any], oracle_model=None) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'] # origin response
        prefer_input_ids = batch['prefer_input_ids']
        prefer_attention_mask = batch['prefer_attention_mask']
        prefer_labels = batch['prefer_labels'] # prefer response
        if 'retainlabels' in batch:
            retainlabels = batch['retainlabels']
        else:
            retainlabels = None
        
        loss, forget_loss, retain_loss = self.calculate_loss(
            model, 
            input_ids, labels, attention_mask,
            prefer_input_ids, prefer_labels, prefer_attention_mask,
            retainlabels,
            oracle_model=oracle_model
        )

        return {
            'loss': loss,
            'forget_loss': forget_loss,
            'retain_loss': retain_loss 
        }

#! Real forget losses 
def GradAscentLossFunc(model, input_ids, attention_mask, labels, **kwargs):
    next_token_loss = NextTokenPredictionLoss(
        model, input_ids, attention_mask, labels
    )
    return -1 * next_token_loss

def GradDescentLossFunc(model, input_ids, attention_mask, labels, **kwargs):
    next_token_loss = NextTokenPredictionLoss(
        model, input_ids, attention_mask, labels
    )
    return next_token_loss

def KLLossFunc(model, input_ids, attention_mask, labels, oracle_model, **kwargs):
    with torch.no_grad():
        retain_outputs = oracle_model(
            input_ids, labels=labels, attention_mask=attention_mask, use_cache=True
        )
        retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
        retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

    outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1)
    probs = probs.view(-1, outputs.logits.shape[-1])
    retain_loss = nn.functional.kl_div(probs, retain_probs, reduction='batchmean', log_target=True)
    return retain_loss

def DPOLossFunc(model, input_ids, labels, label_attention_mask, prefer_input_ids, prefer_labels, prefer_label_attention_mask, oracle_model, beta=0.1, **kwargs):

    with torch.no_grad():
        oracle_origin_outputs = oracle_model(
            input_ids, attention_mask=label_attention_mask, use_cache=True
        )
        oracle_prefer_outputs = oracle_model(
            prefer_input_ids, attention_mask=prefer_label_attention_mask, use_cache=True
        )
        oracle_origin_loss = -1 * TokenNextTokenPredictionLoss(oracle_origin_outputs.logits, labels)
        orcale_prefer_loss = -1 * TokenNextTokenPredictionLoss(oracle_prefer_outputs.logits, prefer_labels)
        
    origin_outputs = model(input_ids, attention_mask=label_attention_mask)
    prefer_outputs = model(prefer_input_ids, attention_mask=prefer_label_attention_mask)
    origin_loss = -1 * TokenNextTokenPredictionLoss(origin_outputs.logits, labels)
    origin_prefer_loss = -1 * TokenNextTokenPredictionLoss(prefer_outputs.logits, prefer_labels)

    pi_logratios = origin_prefer_loss - origin_loss
    ref_logratios = orcale_prefer_loss - oracle_origin_loss

    loss = -F.logsigmoid(
        beta * (pi_logratios - ref_logratios)
    ).mean() * 2 / beta
    return loss

def NPOLossFunc(model, input_ids, attention_mask, labels, oracle_model, beta=0.1, **kwargs):
    with torch.no_grad():
        oracle_outputs = oracle_model(input_ids, attention_mask=attention_mask, use_cache=True)
        oracle_logits = oracle_outputs.logits
        oracle_tokenloss = TokenNextTokenPredictionLoss(oracle_logits, labels)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    tokenloss = TokenNextTokenPredictionLoss(logits, labels)
    log_ratio = tokenloss - oracle_tokenloss

    loss = -F.logsigmoid(beta * log_ratio).mean() * 2 / beta
    return loss

def UniformLossFunc(model, input_ids, attention_mask, labels=None, **kwargs):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    num_labels = logits.shape[-1]
    soft_outputs = nn.functional.softmax(logits, dim=-1).view(-1, num_labels)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / logits.size(-1)).to(logits.device)
    kl_div = torch.nn.functional.kl_div(soft_outputs.log(), uniform_dist, reduction='batchmean')
    return kl_div

def create_unlearn_loss(loss_config):
    if (forget_loss := loss_config.get('forget_loss', None)) is None:
        forget_loss_func = None
    else:
        try:
            forget_loss_func = globals()[forget_loss]
        except KeyError:
            raise NotImplementedError(f"Invalid forget loss: {forget_loss}")
    
    if (retain_loss := loss_config.get('retain_loss', None)) is None:
        retain_loss_func = None
    else:
        try:
            retain_loss_func = globals()[retain_loss]
        except KeyError:
            raise NotImplementedError(f"Invalid forget loss: {retain_loss}")
    
    if forget_loss_func.__name__ == 'DPOLossFunc':
        return PreferLoss(
            forget_loss_func, retain_loss_func, retain_weight=loss_config.get('retain_weight', 1.0)
        ) 
    else:
        return ForgetRetainLoss(
            forget_loss_func, retain_loss_func, retain_weight=loss_config.get('retain_weight', 1.0)
        )

def loss_requries_oracle(loss_config):
    forget_loss = loss_config.get('forget_loss', None)
    retain_loss = loss_config.get('retain_loss', None)
    if retain_loss in [
        "KLLossFunc"
    ]:
        return True

    if forget_loss in [
        "NPOLossFunc",
        "DPOLossFunc",
    ]:
        return True
    return False
