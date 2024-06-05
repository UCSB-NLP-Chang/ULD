import os
import glob
import datetime
import structlog
from codetiming import Timer
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.core.hydra_config import HydraConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from pytorch_lightning import seed_everything

from .log_util import configure_structlog

def NameTimer(name):
    return Timer(name, text="{name} spent: {:0.4f} seconds")

def init_script(hparams):
    #* Initialize all configs and return a structlog LOGGER object
    OmegaConf.resolve(hparams)
    unfilled_paths= find_unfilled_paths(hparams)
    if len(unfilled_paths) > 0:
        err = "\n".join(
            [f"{'.'.join(map(str, path))}" for path in unfilled_paths]
        )
        raise ValueError(f"Unfilled paths in config:\n {err}")
    hydraconf = HydraConfig.get()
    configure_structlog(f"{hydraconf.runtime.output_dir}/{hydraconf.job.name}.log")
    LOGGER = structlog.getLogger()
    return LOGGER

def find_unfilled_paths(conf, path=None):
    if path is None:
        path = []
    paths_with_unfilled = []

    if isinstance(conf, DictConfig):
        for key in conf.keys():
            try:
                value = conf[key]
            except Exception as e:
                value = '???'
            new_path = path + [key]  
            if isinstance(value, (DictConfig, ListConfig)):
                paths_with_unfilled.extend(find_unfilled_paths(value, new_path))  
            elif value == '???':
                paths_with_unfilled.append(new_path)
    elif isinstance(conf, ListConfig):
        for index in range(len(conf)):
            try:
                item = conf[index]
            except Exception as e:
                item = '???'
            new_path = path + [index]
            if isinstance(item, (DictConfig, ListConfig)):
                paths_with_unfilled.extend(find_unfilled_paths(item, new_path))
            elif item == '???':
                paths_with_unfilled.append(new_path)
        
    return paths_with_unfilled

def set_progress(disable=False):
    return Progress(
        TextColumn("[bold blue]{task.fields[name]}", justify="left"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        disable=disable,
    ) 

def create_log_dir(configs):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if configs.name and configs.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if configs.resume:
        if not os.path.exists(configs.resume):
            raise ValueError("Cannot find {}".format(configs.resume))
        if os.path.isfile(configs.resume):
            paths = configs.resume.split("/")
            idx = len(paths) - paths[::-1].index(configs.base_logdir) + 1
            logdir = "/".join(paths[:idx])
            ckpt = configs.resume
        else:
            assert os.path.isdir(configs.resume), configs.resume
            logdir = configs.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        configs.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        configs.base = base_configs + configs.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if configs.name:
            name = configs.name + "/"
        elif configs.base:
            cfg_name = os.path.split(configs.base[0])[-1]
            cfg_name = os.path.splitext(cfg_name)[0]
            name = cfg_name + "_"
        else:
            name = ""
        nowname = name + now + configs.postfix
        if configs.debug:
            logdir = os.path.join(
                configs.base_logdir, configs.project, "debug", nowname)
        else:
            logdir = os.path.join(configs.base_logdir, configs.project, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(configs.seed)

    return now, nowname, logdir, ckptdir, cfgdir