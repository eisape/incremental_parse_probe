import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import yaml
import argparse
import shutil
import random
from pathlib import Path

import torch

import architectures
import datasets
from utils import *

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from experiment import IncrementalParseProbeExperiment

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="filename", default="./configs/test.yaml")
parser.add_argument("--device", dest="device", default=[0], nargs="+")

args = parser.parse_args()
device = [int(d) for d in args.device]
config_path = args.filename
with open(args.filename, "r") as file:
    args = yaml.safe_load(file)

print(f"======= Training {args['probe_params']['probe_name']} =======")

args["trainer_params"]["gpus"] = device
args["exp_params"]["manual_seed"] = random.randint(1000, 2000)

tb_logger = TensorBoardLogger(
    save_dir=args["logging_params"]["save_dir"],
    name=args["probe_params"]["probe_name"],
    version=args["logging_params"]["version"],
)
Path(f"{tb_logger.log_dir}").mkdir(exist_ok=True, parents=True)
shutil.copy2(config_path, f"{tb_logger.log_dir}/config.yaml")
seed_everything(args["exp_params"]["manual_seed"], True)

args["probe_params"]["pretrained_model"] = args["pretrained_model"]

probe = getattr(architectures, args["probe_params"]["probe_type"])(
    args["probe_params"]
).to("cuda")

Trainer(
    logger=tb_logger,
    callbacks=[
        EarlyStopping(monitor="val_loss"),
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=5,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}",
            save_last=True,
        ),
    ],
    strategy="ddp",
    **args["trainer_params"],
).fit(
    IncrementalParseProbeExperiment(probe=probe, params=args["exp_params"]),
    datamodule=datasets.PTB_Dataset(config=args, probe=probe),
)
