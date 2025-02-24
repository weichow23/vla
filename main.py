import os
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
from re import L
from typing import Dict, Any
import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
import torch.distributed as dist

from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.datamodule.gr_datamodule import GRDataModule
from robovlms.data.data_utils import preprocess_image
from robovlms.utils.setup_callback import SetupCallback


def get_date_str():
    return str(datetime.date.today())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def init_lr_monitor_callback():
    return LearningRateMonitor(logging_interval="step")


def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(" ", "_"),
        logdir=config["log_dir"],
        ckptdir=config["output_dir"],
        cfgdir=config["log_dir"],
        config=config,
    )


def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs["trainer"])
    trainer_config["devices"] = configs.get("gpus", "auto")
    trainer_config["num_nodes"] = configs.get("num_nodes", 1)
    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    exp_name = configs.get("exp_name", "default")

    if "strategy" not in trainer_config or trainer_config["strategy"] == "ddp":
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)

    # init loggers
    loggers = None
    log_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["log_dir"] = configs["log_root"] / log_dir
    if isinstance(trainer_config.get("logger"), list):
        loggers = []
        for logger in trainer_config.get("logger"):
            if logger == "tensorboard":
                loggers.append(
                    TensorBoardLogger(configs["log_dir"].as_posix(), name=exp_name)
                )
            elif logger == "csv":
                loggers.append(CSVLogger(configs["log_dir"].as_posix(), name=exp_name))
            else:
                raise NotImplementedError

    trainer_config["logger"] = loggers

    ckpt_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["output_dir"] = configs["output_root"] / ckpt_dir

    configs["log_dir"].mkdir(parents=True, exist_ok=True)
    configs["output_dir"].mkdir(parents=True, exist_ok=True)
    configs["cache_root"].mkdir(parents=True, exist_ok=True)
    # os.system(f"sudo chmod 777 -R runs/")

    configs["log_dir"] = configs["log_dir"].as_posix()
    configs["output_dir"] = configs["output_dir"].as_posix()
    configs.pop("output_root")
    configs.pop("log_root")
    configs["cache_root"] = configs["cache_root"].as_posix()

    trainer_config["callbacks"] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(dirpath=configs["output_dir"], save_top_k=-1, every_n_epochs=1),
    ]

    return trainer_config


def experiment(variant):
    seed_everything(variant["seed"] + int(os.environ["RANK"]))
    # import pdb; pdb.set_trace()
    trainer_config = init_trainer_config(variant)
    model_load_path = variant.get("model_load_path", None)

    trainer = Trainer(**trainer_config)
    variant["gpus"] = trainer.num_devices
    variant["train_setup"]["precision"] = variant["trainer"]["precision"]

    if variant["fwd_head"] is not None:
        variant["train_setup"]["predict_forward_hand"] = variant["fwd_head"].get(
            "pred_hand_image", False
        )

    if not os.path.exists(variant['model_path']):
        repo_name = variant["model_url"].split("/")[-1].split(".")[0]
        print(
            f"VLM backbone does not exist, cloning {variant['model']} from {variant['model_url']}..."
        )
        os.system(f"git clone {variant['model_url']} .vlms/{repo_name}")
        variant['model_path'] = ".vlms/" + repo_name
        variant['model_config'] = os.path.join(variant['model_path'], "config.json")
    
    if variant["model"] == "kosmos":
        import transformers

        package_dir = transformers.__path__[0]
        os.system(
            "cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(
                package_dir
            )
        )

        import importlib

        importlib.reload(transformers)
    
    model = BaseTrainer.from_checkpoint(
        model_load_path, variant.get("model_load_source", "torch"), variant
    )

    image_preprocess = model.model.image_processor

    _kwargs = {
        "model": model,
        "datamodule": GRDataModule(
            variant["train_dataset"],
            variant["val_dataset"],
            variant["batch_size"],
            variant["num_workers"],
            tokenizer=model.model.tokenizer,
            tokenizer_config=variant["tokenizer"],
            fwd_pred_next_n=variant["fwd_pred_next_n"],
            window_size=variant["window_size"],
            image_size=variant["image_size"],
            image_fn=functools.partial(
                preprocess_image,
                image_processor=image_preprocess,
                model_type=variant["model"],
            ),
            discrete=(
                False
                if variant["act_head"] is None
                else variant["act_head"].get("action_space", "continuous") == "discrete"
            ),
            discrete_action=(
                False
                if variant["act_head"] is None
                else variant["act_head"].get("action_space", "continuous") == "discrete"
            ),
            use_mu_law=variant.get("use_mu_law", False),
            mu_val=variant.get("mu_val", 255),
            n_bin=(
                256
                if variant["act_head"] is None
                else variant["act_head"].get("n_bin", 256)
            ),
            min_action=(
                -1
                if variant["act_head"] is None
                else variant["act_head"].get("min_action", -1)
            ),
            max_action=(
                1
                if variant["act_head"] is None
                else variant["act_head"].get("max_action", 1)
            ),
            discrete_action_history=variant.get("discrete_action_history", False),
            act_step=variant.get("fwd_pred_next_n", 1),
            norm_action=variant.get("norm_action", False),
            norm_min=variant.get("norm_min", -1),
            norm_max=variant.get("norm_max", 1),
            regular_action=variant.get("regular_action", False),
            x_mean=variant.get("x_mean", 0),
            x_std=variant.get("x_std", 1),
            weights=variant.get("train_weights", None),
            tcp_rel=variant.get("tcp_rel", False),
            # vit_name=vit_name,
            model_name=variant.get("model", "flamingo"),
        ),
        "ckpt_path": variant["resume"],
    }
    if _kwargs["ckpt_path"] is not None:
        print(f"Resuming from {variant['resume']}...")

    trainer.fit(**_kwargs)


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def update_configs(configs, args):
    configs["raw_config_path"] = args["config"]
    configs["output_root"] = (
        Path(configs["output_root"]) / configs["model"] / configs["task_name"]
    )
    configs["log_root"] = (
        Path(configs["log_root"]) / configs["model"] / configs["task_name"]
    )
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    for k, v in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                # assert sub_k in configs[k], f"{sub_k} not in configs {k}"
                if sub_v != None:
                    configs[k][sub_k] = sub_v
        else:
            if v != None:
                configs[k] = v
    return configs


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("config", type=str, help="config file used for training")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--annotation_file", default=None, type=str)
    parser.add_argument("--model_load_path", default=None, type=str)
    parser.add_argument("--data_subfolder", default=None, type=str)
    parser.add_argument("--task_num", default=None, type=int)
    parser.add_argument("--seq_len", default=None, type=float)
    parser.add_argument("--exp_name", default=None, type=str)

    # Loss
    parser.add_argument("--arm_gripper_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_pred_next_n", default=None, type=int)

    parser.add_argument("--use_multi_modal_emb", default=False, action="store_true")
    parser.add_argument(
        "--no_video_pretrained_model", default=False, action="store_true"
    )
    parser.add_argument("--finetune", default=False, action="store_true")

    # Training
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--min_lr_scale", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=int)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)

    global_names = set(vars(parser.parse_known_args()[0]).keys())

    # Trainer
    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--strategy", default=None, type=str)
    trainer_parser.add_argument("--precision", default=None, type=str)
    trainer_parser.add_argument("--gradient_clip_val", default=None, type=float)
    trainer_parser.add_argument("--max_epochs", default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    # Model Architecture
    llm_parser = parser.add_argument_group("llm")
    llm_parser.add_argument("--type", default=None, type=str)
    llm_parser.add_argument("--n_embd", default=None, type=int)
    llm_parser.add_argument("--n_layer", default=None, type=int)
    llm_parser.add_argument("--n_head", default=None, type=int)
    llm_names = (
        set(vars(parser.parse_known_args()[0]).keys()) - global_names - trainer_names
    )

    args = {}
    trainer_args = {}
    llm_args = {}
    temp_args = vars(parser.parse_args())
    for k, v in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v
        elif k in llm_names:
            llm_args[k] = v

    args["llm"] = llm_args
    args["trainer"] = trainer_args

    return args


if __name__ == "__main__":
    # import os

    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    args = parse_args()

    # load config files
    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)

    dist.init_process_group(backend="nccl")
    experiment(variant=configs)
