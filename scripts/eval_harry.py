import os
import hydra
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import copy
from codetiming import Timer

from uld.utils import init_script
from uld.data.conv_util import create_template
from uld.model import EVAL_INIT_FUNCS
from uld.harryutil.convert_data import RETAIN_TASKS, get_wikitext, get_HPQA
from uld.harryutil.evaluate_util import eval_copyright_leakage, eval_retain_ppl, eval_retain_qa

@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(configs):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    device = f'cuda:{configs.gpu.gpu_id}'
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

    path = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.model_max_length = model.config.max_position_embeddings
    tokenizer.padding_side = "left"
    tokenizer.padding_size = 'longest'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    right_pad_tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.model_max_length = model.config.max_position_embeddings
    right_pad_tokenizer.padding_side = 'right'
    right_pad_tokenizer.padding_size = 'longest'
    if right_pad_tokenizer.pad_token is None:
        right_pad_tokenizer.pad_token = tokenizer.eos_token

    full_results = {}
    with Timer("Evaluation", text="{name} spent: {:0.4f} seconds"):
        #! Copyrighted text
        hpdata = get_HPQA('hp_train_qa_200', num=100)
        eval_res = eval_copyright_leakage(OUTPUTDIR, LOGGER, 'hp-prefix200', hpdata, conv_template, model, tokenizer)
        full_results.update(eval_res)

        #! Retain qas
        retain_conv_template = copy.deepcopy(conv_template)
        retain_conv_template.question_start_token = ""
        retain_conv_template.question_end_token = " "
        retain_conv_template.answer_token = ""

        #! Retain ppl
        eval_data = get_wikitext()
        eval_res = eval_retain_ppl(OUTPUTDIR, LOGGER, 'wikitext', eval_data, retain_conv_template, model, right_pad_tokenizer)
        full_results.update(eval_res)

        for task_name, task_func in RETAIN_TASKS.items():
            eval_data = task_func()
            num_choices = len(eval_data[0]['choices'])
            eval_res = eval_retain_qa(OUTPUTDIR, LOGGER, task_name, eval_data, retain_conv_template, model, right_pad_tokenizer, num_choices)
            full_results.update(eval_res)

        accs = [v for k, v in full_results.items() if 'acc' in k]
        full_results.update({"mean-acc": np.mean(accs).item()})

    df = pd.DataFrame.from_records([full_results])
    df.to_csv(os.path.join(OUTPUTDIR, "aggregate-stat.csv"), index=False)


if __name__ == "__main__":
    main()