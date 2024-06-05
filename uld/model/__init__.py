from peft import PeftModel
from transformers import AutoConfig

from .contrastllm import ContrastLLM
from .offsetllm import create_offset_model
from .utils import *
from ..utils import NameTimer

TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
}

def eval_create_base_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Base model"):
        if os.path.exists(os.path.join(ckpt_path, 'adapter_config.json')):
            #! A lora model
            if os.path.exists(os.path.join(ckpt_path, '../fullmodel')):
                # small assistant
                base_path = os.path.join(ckpt_path, '../fullmodel')
            else:
                base_path = base_model_config.model_path
            model = AutoModelForCausalLM.from_pretrained(
                base_path, torch_dtype=torch.bfloat16
            ).to(device)
            peftmod = PeftModel.from_pretrained(
                model, ckpt_path, torch_dtype=torch.bfloat16
            )
            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(device)
            return peftmod 
        else:
            # Base only
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16
            ).to(device)
            return model


def eval_create_uld_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading ULD model"):
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, torch_dtype=torch.bfloat16
        ).to(device)
        with NameTimer("Loading assistant"):
            small_full_path = os.path.join(ckpt_path, '../fullmodel')
            assistant = AutoModelForCausalLM.from_pretrained(
                small_full_path, torch_dtype=torch.bfloat16
            ).to(device)
            peftmod = PeftModel.from_pretrained(
                assistant, ckpt_path, torch_dtype=torch.bfloat16
            )
            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(device)
        
        model = ContrastLLM(
            basellm, peftmod, 
            weight=model_mode_config.weight, 
            top_logit_filter=model_mode_config.top_logit_filter,
        ) 
        return model

def eval_create_offset_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Offset model"):
        config = AutoConfig.from_pretrained(ckpt_path)
        if hasattr(config, 'is_offset') and config.is_offset:
            if hasattr(config, 'weight'):
                weight = config.weight
            else:
                weight = 1.0
            base_name = config.base_model_name
            model = create_offset_model(
                base_name, 
                device=device, 
                base_assist_path=config.base_assist_path, 
                weight=weight, 
                new_assist_path=ckpt_path
            )
            return model

TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
}

EVAL_INIT_FUNCS = {
    "base": eval_create_base_model,
    "uld": eval_create_uld_model,
    "offset": eval_create_offset_model,
}