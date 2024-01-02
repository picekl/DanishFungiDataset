import os
import logging
import torch
import wandb
from typing import Tuple

import pandas as pd
import torch.nn as nn

from scipy.special import softmax
from torch.utils.data import DataLoader

from fgvc.core.training import train, predict
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import (
    get_optimizer_and_scheduler,
    load_args,
    load_config,
    load_model,
    load_train_metadata,
    save_config,
)
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import (
    finish_wandb,
    init_wandb,
    resume_wandb,
    set_best_scores_in_summary,
)

from hfhub import export_model_to_huggingface_hub_from_checkpoint

logger = logging.getLogger("script")

SCRATCH_DIR = os.getenv("SCRATCHDIR", "/media/Data-10T-1/Data/")
SHARED_SCRATCH_DIR = "/scratch.shared/picekl"
# API_BASE_PATH = "http://147.228.47.72:12080/files/"


def load_metadata(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata of the traning and validation sets."""
    assert "dataset" in config

    train_df = pd.read_csv(f"../metadata/DanishFungi2023-train_mini.csv")  # TODO .. > .

    valid_df = pd.read_csv(f"../metadata/DanishFungi2023-val_mini.csv")  # TODO

    train_df["image_path"] = train_df.image_path.apply(
        lambda path: os.path.join(SHARED_SCRATCH_DIR, path))

    valid_df["image_path"] = valid_df.image_path.apply(
        lambda path: os.path.join(SHARED_SCRATCH_DIR, path))

    return train_df, valid_df


def evaluate(
    model: nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
    path: str,
    device: torch.device = "cpu",
):
    """Evaluate model and create example visualizations.

    Parameters
    ----------
    model
        Model to evaluate.
    validloader
        Validation dataloader.
    path
        Directory to store example visualizations.
    device
        Cuda or CPU device.
    """
    if wandb.run is None:
        return

    # evaluate model
    logger.info("Creating predictions.")
    preds, targs, _, scores = predict(model, validloader, device=device)
    argmax_preds = preds.argmax(1)
    max_conf = softmax(preds, 1).max(1)

    # create wandb prediction table
    train_df = trainloader.dataset.df
    valid_df = validloader.dataset.df
    id2class = dict(zip(train_df["class_id"], train_df["species"]))

    pred_df = pd.DataFrame()
    # pred_df["image"] = (
    #     valid_df["image_path"]
    #     .str.replace(SCRATCH_DIR, API_BASE_PATH, regex=False)
    #     .apply(lambda x: wandb.Image(data_or_path=x))
    # )
    pred_df["species"] = valid_df["species"]
    pred_df["species-predicted"] = [id2class[x] for x in argmax_preds]
    # pred_df["class_id"] = valid_df["class_id"]
    # pred_df["class_id-predicted"] = argmax_preds
    pred_df["max_confidence"] = max_conf
    for col in ["image_path"]:
        pred_df[col] = valid_df[col]
    wandb.log({"pred_table": wandb.Table(dataframe=pred_df)})

    
def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> dict:
    """Include information from metadata to the training configuration."""
    assert "class_id" in train_df and "class_id" in valid_df
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(valid_df)
    return config


def train_clf(
    *,
    train_metadata: str = None,
    valid_metadata: str = None,
    config_path: str = None,
    cuda_devices: str = None,
    wandb_entity: str = None,
    wandb_project: str = None,
    resume_exp_name: str = None,
    **kwargs,
):
    """Train model on the classification task."""
    if train_metadata is None or valid_metadata is None or config_path is None:
        # load script args
        args, extra_args = load_args()
        config_path = args.config_path
        cuda_devices = args.cuda_devices
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
        resume_exp_name = args.resume_exp_name
    else:
        extra_args = kwargs

    # load training config
    logger.info("Loading training config.")
    config = load_config(
        config_path,
        extra_args,
        run_name_fmt="architecture-loss-augmentations",
        resume_exp_name=resume_exp_name,
    )

    # set device and random seed
    device = set_cuda_device(cuda_devices)
    set_random_seed(config["random_seed"])

    # load metadata
    logger.info("Loading training and validation metadata.")
    train_df, valid_df = load_metadata(config)
    config = add_metadata_info_to_config(config, train_df, valid_df)
    
    # load model and create optimizer and lr scheduler
    logger.info("Creating model, optimizer, and scheduler.")
    model, model_mean, model_std = load_model(config)

    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    # create dataloaders
    logger.info("Creating DataLoaders.")
    trainloader, validloader, _, _ = get_dataloaders(
        train_df,
        valid_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # create loss function
    logger.info("Creating loss function.")
    if config["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "FocalLoss":
        criterion = FocalLossWithLogits()
    elif config["loss"] == "SeeSawLoss":
        class_counts = train_df["class_id"].value_counts().sort_index().values
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        logger.error(f"Unknown loss: {config['loss']}")
        raise ValueError()

    # init wandb
    if wandb_entity is not None and wandb_project is not None:
        if resume_exp_name is None:
            init_wandb(
                config, config["run_name"], entity=wandb_entity, project=wandb_project
            )
        else:
            if "wandb_run_id" not in config:
                raise ValueError("Config is missing 'wandb_run_id' field.")
            resume_wandb(
                run_id=config["wandb_run_id"],
                entity=wandb_entity,
                project=wandb_project,
            )

    # save config to json in experiment path
    if resume_exp_name is None:
        save_config(config)

    # train model
    logger.info("Training the model.")
    train(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        accumulation_steps=config.get("accumulation_steps", 1),
        clip_grad=config.get("clip_grad"),
        device=device,
        seed=config.get("random_seed", 777),
        path=config["exp_path"],
        resume=resume_exp_name is not None,
        mixup=config.get("mixup"),
        cutmix=config.get("cutmix"),
        mixup_prob=config.get("mixup_prob"),
        apply_ema=config.get("apply_ema"),
        ema_start_epoch=config.get("ema_start_epoch", 0),
        ema_decay=config.get("ema_decay", 0.9999),
    )

        
    # evaluate model
    model_filename = os.path.join(config["exp_path"] + "/best_f1.pth")
    model.load_state_dict(torch.load(model_filename, map_location="cpu"))
    evaluate(model, trainloader, validloader, path=config["exp_path"], device=device)

    
    # finish wandb run
    run_id = finish_wandb()
    if run_id is not None:
        logger.info("Setting the best scores in the W&B run summary.")
        set_best_scores_in_summary(
            run_or_path=f"{wandb_entity}/{wandb_project}/{run_id}",
            primary_score="Val. F1",
            scores=lambda df: [col for col in df if col.startswith("Val.")],
        )
    try:
        config["mean"] = model_mean
        config["std"] = model_std
        export_model_to_huggingface_hub_from_checkpoint(
            config=config,
            repo_owner="BVRA",
            saved_model="f1"
        )
    except Exception as e:
        print(f"Exception during export: {e}")
        

if __name__ == "__main__":
    train_clf()
