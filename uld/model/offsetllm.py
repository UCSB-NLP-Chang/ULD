from typing import Optional, List
import torch
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    GenerationConfig, 
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model

from .peft_util import find_all_linear_names

#! Unofficial implementation for the paper 'Offset Unlearning For Large Language Model' (https://arxiv.org/pdf/2404.11045)
def create_offset_model(model_path, data_type='bfloat16', **kwargs):
    baseconfig = AutoConfig.from_pretrained(model_path)
    model = OffsetAssitedModel(
        baseconfig, torch_dtype=torch.bfloat16, **kwargs,
    )
    if device := (kwargs.get('device', None)):
        model = model.to(device=device)
    return model

class OffsetAssitedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"assist_model.*",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"assist_model.*",
    ]

    def __init__(self, config, base_assist_path, new_assist_path=None, weight=1.0, is_lora=False, Lora=OmegaConf.create({"r":0, "alpha": 32, "dropout": 0.05}), **kwargs):
        tmplora = OmegaConf.to_container(Lora)
        config.Lora = tmplora
        config.base_model_name = config._name_or_path
        config.is_offset = True
        config.base_assist_path = base_assist_path
        config.new_assist_path = new_assist_path
        config.weight = weight
        config.new_assist_path = new_assist_path
        super().__init__(config, **kwargs)
        
        self.vocab_size = config.vocab_size

        self.basellm = AutoModelForCausalLM.from_pretrained(config.base_model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        self.basellm.eval()
        self.basellm.requires_grad_(False) #! Freeze

        self.base_assist_llm = AutoModelForCausalLM.from_pretrained(base_assist_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16)
        self.base_assist_llm.eval()
        self.base_assist_llm.requires_grad_(False) #! Freeze
        
        if new_assist_path is None:
            assist_path = base_assist_path
        else:
            assist_path = new_assist_path
        
        self.assist_llm = AutoModelForCausalLM.from_pretrained(assist_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16)
        if Lora.r != 0:
            peftconfig = LoraConfig(
                r=Lora.r,
                lora_alpha=Lora.alpha,
                target_modules=find_all_linear_names(self.assist_llm), 
                lora_dropout=Lora.dropout,
                bias=Lora.bias, 
                task_type="CAUSAL_LM",
            )
            self.assist_llm = get_peft_model(self.assist_llm, peftconfig)

        self.weight = weight
        self.generation_config = GenerationConfig.from_model_config(self.config)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.basellm.prepare_inputs_for_generation(
            *args, **kwargs
        )

    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.no_grad():
            outputs = self.basellm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # cache_position=cache_position,
            )
            base_logits = outputs.logits.detach() # make sure the gradient stops for oracle
            outputs = self.base_assist_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            base_assist_logits = outputs.logits.detach()

        assist_outputs = self.assist_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        assist_logits = assist_outputs.logits

        logits = base_logits + self.weight * (assist_logits - base_assist_logits) #! ajust the final distribution
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def save_pretrained(self, path, **kwargs):
        self.assist_llm.save_pretrained(path)
        self.config.save_pretrained(path)
