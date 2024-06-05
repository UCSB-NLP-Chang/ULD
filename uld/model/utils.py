import os
import copy 
import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from ..utils import NameTimer
from .peft_util import find_all_linear_names

def get_dtype(data_type):
    if data_type == 'bfloat16':
        return torch.bfloat16
    elif data_type == 'float16':
        return torch.float16

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def copy_weights(base_llm, model):
    config = model.config
    name = model.config._name_or_path.lower()
    if ('llama' in name) or ('zephyr' in name) or ('mistral' in name):
        print(f"Copying {name} first layer: {config.num_hidden_layers}")
        model.model.embed_tokens.load_state_dict(
            base_llm.model.embed_tokens.state_dict()
        )
        model.model.norm.load_state_dict(
            base_llm.model.norm.state_dict()
        )
        for layer_num in range(config.num_hidden_layers):
            model.model.layers[layer_num].load_state_dict(
                base_llm.model.layers[layer_num].state_dict()
            )
        model.lm_head.load_state_dict(
            base_llm.lm_head.state_dict()
        )
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")

def init_small_llm(origin_config, num_layer, device, hparams=None, base_llm=None, saved_path=None):
    config = copy.deepcopy(origin_config)
    config.num_hidden_layers = num_layer
    model = AutoModelForCausalLM.from_config(
        config,
        use_flash_attention_2=True, 
        torch_dtype=torch.bfloat16, 
    ).to('cpu')

    if base_llm is not None:
        copy_weights(base_llm, model)
        
    if saved_path is not None:
        model.load_state_dict(
            torch.load(saved_path)
        )

    return model

def create_full_model(model_path, num_layer=0 ,data_type='bfloat16', **kwargs):
    with NameTimer("Init full model"):
        basellm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=get_dtype(data_type),
            use_flash_attention_2=True, trust_remote_code=True,
        )
        if num_layer != 0: #! Construct the small model
            basellm = init_small_llm( 
                basellm.model.config,
                num_layer=num_layer,
                base_llm=basellm,
                device='cpu',
            )
        return basellm

def create_peft_model(model_path, Lora, baseoutdir, num_layer=0, data_type='bfloat16', **kwargs):
    with NameTimer("Init peft model"):
        basellm = create_full_model(model_path, num_layer, data_type)
        if num_layer != 0:
            #! We save the extracted small LLM to disk to speed up test time model loading
            basellm.save_pretrained(os.path.join(baseoutdir, 'fullmodel'))
        peftconfig = LoraConfig(
            r=Lora.r,
            lora_alpha=Lora.alpha,
            target_modules=find_all_linear_names(basellm), 
            lora_dropout=Lora.dropout,
            bias=Lora.bias, 
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(basellm, peftconfig)
        return model
