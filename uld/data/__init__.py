from omegaconf import OmegaConf
from .tofu import ToFU_DataModule
from .harry import HarryPotterDataModule

def create_datamod(dataset_config, conv_template_config, data_mode_config, tokenizer=None, **kwargs):
    print(dataset_config)
    class_name = dataset_config.get('class_name', None)
    if "ToFU".lower() in class_name.lower():
        mod = ToFU_DataModule
    elif 'Harry'.lower() in class_name.lower():
        mod = HarryPotterDataModule
    else:
        raise ValueError("Unkown data module class")

    return mod(
        tokenizer=tokenizer,
        conv_template_config=conv_template_config,
        **dataset_config,
        **data_mode_config,
        **kwargs,
    )
