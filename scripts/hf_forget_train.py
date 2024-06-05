#! This script initializes a small LLM and finetune to remember some facts
import os
import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from uld.utils import init_script, create_log_dir, NameTimer
from uld.data import create_datamod

from uld.model import TRAIN_INIT_FUNCS
from uld.model.forget_losses import create_unlearn_loss, loss_requries_oracle
from uld.hfutil import ForgetTrainer, SimpleProfileCallback
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

@hydra.main(version_base=None, config_path="../configs", config_name="tune_config")
def main(configs):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    # ! Setup Logger
    BASELOGDIR = configs.BASELOGDIR
    output_dir = HydraConfig.get().runtime.output_dir
    configs.base_logdir = os.path.join(output_dir, "logs")
    LOGGER = init_script(configs)
    LOGGER.info("Config", configs=configs)
    LOGGER.info(f"num_devices: {num_devices}")

    OmegaConf.set_struct(configs, False)  # Disable struct mode temporarily
    all_choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    configs.name = "|".join([
        "dataset:" + all_choices.get('data'),
        "loss:" + all_choices.get('unlearn_loss'), 
        "model:" + all_choices.get('model'),
        "datamode:" + all_choices.get('data_mode'), 
    ])
    print("RunName", configs.name)
    OmegaConf.set_struct(configs, True)

    now, nowname, logdir, ckptdir, cfgdir = create_log_dir(configs)
    os.makedirs(logdir, exist_ok=True)
    
    #! setup dataset
    model_config = configs.model
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
    tokenizer.padding_side = "right"
    if "mistral" in model_config.model_name.lower():
        tokenizer.padding_side = "left" #! no idea why this is needed for mistral
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_module = create_datamod(
        dataset_config=configs.data.dataset,
        conv_template_config=configs.data.conv_template,
        data_mode_config=configs.data_mode,
        tokenizer=tokenizer,
    )
    data_module.prepare_data()
    data_module.setup('fit')

    trainer_config = configs.get("trainer", OmegaConf.create())    
    batch_size = configs.trainer.batch_size
    train_data_size = len(data_module.train_dataloader()) * batch_size
    num_update_steps_per_epoch = train_data_size // (num_devices * batch_size * trainer_config.gradient_accumulation_steps)
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps = num_update_steps_per_epoch * trainer_config.max_epochs 
    print("num_training_steps", num_training_steps)

    #! change checkpoint foler at runtime
    tmpckptdir = ckptdir.split(BASELOGDIR)[-1]
    checkpoint_dir = os.path.join(
        configs.OUTPUTMODELDIR, "/".join(tmpckptdir.split("/")[1:-1]).replace(",", "|").replace("=","_")
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_config = configs.get('model')
    is_offset = 'offset' in model_config.get('mode', 'base')

    #! setup trainer
    os.makedirs(logdir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = configs.project
    os.environ["WANDB_DIR"] = logdir
    is_deepspeed = 'deepspeed' in trainer_config.get('strategy', "")
    if is_deepspeed:
        print("Loading deepspeed")
        deepspeed_configfile = "configs/ds_config.json"
    else:
        print("None deepspeed")
        deepspeed_configfile = None

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
        warmup_steps=int(num_training_steps * trainer_config.warmup_ratio),
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        max_steps=num_training_steps,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=1,
        logging_dir=logdir,
        output_dir=checkpoint_dir,
        optim="paged_adamw_32bit",
        save_only_model=True,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_configfile,
        save_steps=num_update_steps_per_epoch,
        eval_steps=num_update_steps_per_epoch,
        evaluation_strategy="steps",
        seed=configs.get('seed', 42),
        report_to='wandb',
        run_name=configs.name,
        remove_unused_columns=False,
    )
    
    simpleprofilercallback = SimpleProfileCallback(
        logdir, "simpleprofile.txt"
    )

    #! Logging training mode
    batch = next(iter(data_module.train_dataloader()))
    sampledatas = {
        "train_sample_keys": batch.keys(),
        "train_sample": tokenizer.batch_decode(batch['input_ids'][:2], skip_special_tokens=True),
    }
    if 'prefer_input_ids' in batch:
        sampledatas['prefer_sample'] = tokenizer.batch_decode(batch['prefer_input_ids'][:2], skip_special_tokens=True)
    if 'retainlabel' in batch:
        sampledatas['retainlabel'] = batch['retainlabel'].tolist()
    LOGGER.info("Sample data", **sampledatas, shape=batch['input_ids'].shape)

    train_set = data_module.train_set()
    val_set = data_module.val_set()

    #! Setup model
    baseoutdir = checkpoint_dir
    model_mode = configs.get('model_mode', None)
    init_func = TRAIN_INIT_FUNCS.get(model_mode.get('mode', 'base'))
    model = init_func(
        **model_config,
        **model_mode,
        baseoutdir=baseoutdir,
    )
    model_path = model_config.get('model_path')
    model = model.train()

    #! Setup loss function
    loss_config = configs.get('unlearn_loss')
    loss_function = create_unlearn_loss(loss_config)
    if loss_requries_oracle(loss_config):
        with NameTimer("Load oracle"):
            oracle_model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                use_flash_attention_2=True, trust_remote_code=True, 
            )
            oracle_model.eval()
            oracle_model.requires_grad_(False)
    else:
        oracle_model = None

    requires_equal_sampler = (loss_function.retain_loss_func is not None) 
    LOGGER.info("Training with equal sampler: ", requires_equal_sampler=requires_equal_sampler)

    custom_callbacks = [simpleprofilercallback]
    trainer = ForgetTrainer(
        model=model,
        train_loss_function=loss_function,
        oracle_model=oracle_model,
        equal_sampler=requires_equal_sampler,
        is_deepspeed=is_deepspeed,
        train_dataset=train_set,
        eval_dataset=val_set,
        seed=configs.get('seed', 42),
        callbacks=custom_callbacks,
        args=training_args,
        is_offset=is_offset,
    )
    model.config.use_cache = False
    trainer.train()

    if local_rank == 0:
        os.symlink(output_dir, os.path.join(checkpoint_dir, "trainlogdir"))

if __name__ == "__main__":
    main()