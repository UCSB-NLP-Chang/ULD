import hydra
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer

from uld.utils import init_script
from uld.data.conv_util import create_template
from uld.tofuutil import tofu_eval
from uld.model import EVAL_INIT_FUNCS
from codetiming import Timer

@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(configs):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    device = f'cuda'
    print("DEVICE", device)

    conv_template = create_template(configs.data.conv_template)
    model_mode = configs.get('model_mode', None)
    init_func = EVAL_INIT_FUNCS.get(model_mode.get('mode', 'base'))
    model = init_func(
        base_model_config=configs.model,
        model_mode_config=configs.model_mode,
        ckpt_path=configs.ckpt_path,
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
    tokenizer.padding_side = "left"
    tokenizer.padding_size = 'longest'
    tokenizer.pad_token = tokenizer.eos_token

    right_pad_tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
    right_pad_tokenizer.padding_side = 'right'
    right_pad_tokenizer.padding_size = 'longest'
    right_pad_tokenizer.pad_token = tokenizer.eos_token

    with Timer("Evaluation", text="{name} spent: {:0.4f} seconds"):
        tofu_eval(OUTPUTDIR, LOGGER, configs.data, model, tokenizer, right_pad_tokenizer, conv_template, only_forget_quality=False)

if __name__ == "__main__":
    main()