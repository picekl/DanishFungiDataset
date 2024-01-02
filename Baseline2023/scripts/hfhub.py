import os
import os.path as osp
import torch
import logging
import warnings
import argparse
import json
import yaml

from functools import wraps
from copy import deepcopy


try:
    import huggingface_hub

    assert hasattr(huggingface_hub, "__version__")  # verify package import not local dir
    HuggingFaceAPI = huggingface_hub.HfApi
    HFHubCreateRepo = huggingface_hub.create_repo
except (ImportError, AssertionError):
    huggingface_hub = None
    HuggingFaceAPI = None
    HFHubCreateRepo = None

logger = logging.getLogger("fgvc")


def if_huggingface_hub_is_installed(func: callable):
    """A decorator function that checks if the HuggingFaceHub library is installed."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if huggingface_hub is not None:
            return func(*args, **kwargs)
        else:
            warnings.warn("Library huggingface_hub is not installed.")

    return decorator


# Used to match the saved model names
SAVED_MODEL_NAMES = {
    "accuracy": "best_accuracy",
    "f1": "best_f1",
    "loss": "best_loss",
    "recall": "best_recall@3",
    "last_epoch": "epoch"
}


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


@if_huggingface_hub_is_installed
def export_model_to_huggingface_hub_from_checkpoint(
        *,
        config: dict = None,
        repo_owner: str = None,
        saved_model: str = None,
        model_card: str = None,
) -> str:
    """
    Exports a saved model to the HuggingFace Hub.
    Creates a new model repo if it does not exist. If it does exist,
    the pytorch_model.bin and config.json files will be overwritten.

    Parameters
    ----------
    config
        A dictionary with experiment configuration. Must have "exp_path" (directory with the FGVC run), "architecture",
         "image_size", and "number_of_classes" as valid keys.
    repo_owner
        The "shortcut" of the HuggingFace repository owner name (owner_name/repository_name).
    saved_model
        String key to select the saved model to export (accuracy, f1, loss, recall, last_epoch).
        best_accuracy.pth is the default.
    model_card
        Description of the model that will be displayed in the HuggingFace Hub (README.md).

    Returns
    -------
    repo_name
        The whole HuggingFace repository name suitable to download the model through timm.
    """
    # load script args

    config = deepcopy(config)
    exp_path = config.get("exp_path")

    api = HuggingFaceAPI()

    saved_model_type_name = SAVED_MODEL_NAMES.get(saved_model, "best_accuracy")

    # create new experiment directory or use existing one
    file_names = os.listdir(exp_path)

    model_path = None
    for file_name in file_names:
        if file_name.endswith(".pth") and saved_model_type_name in file_name:
            model_path = osp.join(exp_path, file_name)
            break
    assert osp.exists(model_path), f"Model path {model_path} does not exist."

    # Save selected model saves as bin
    model = torch.load(model_path)
    print(model_path)
    model_bin_path = f'{remove_suffix(model_path, ".pth")}.bin'

    logging.info(f"Saving model to {model_bin_path}")

    torch.save(model, model_bin_path)

    fgvc_config_path = osp.join(exp_path, "config.yaml")
    if len(config) == 1:  # Try to load config if only the exp_path is given
        assert osp.exists(fgvc_config_path), f"Config path {fgvc_config_path} does not exist."
        with open(fgvc_config_path, 'r') as fp:
            config_data = yaml.safe_load(fp)
            config.update(config_data)

    timm_config_path = osp.join(exp_path, "config.json")
    _create_timm_config(config, timm_config_path)

    repo_name = _create_model_repo_name(repo_owner, config)

    logging.info(f"Creating new repository: {repo_name}")

    # Get mean, std
    if "mean" not in config or "std" not in config:
        config["mean"] = tuple(model.default_cfg["mean"])
        config["std"] = tuple(model.default_cfg["std"])

    if model_card is None:
        model_card = get_default_model_card(config, repo_name)

    # Create model card file
    model_card_path = create_model_card_file(model_card, exp_path)

    try:
        HFHubCreateRepo(repo_id=repo_name, repo_type="model", exist_ok=True)
        # Upload model
        api.upload_file(
            path_or_fileobj=model_bin_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_name,
            repo_type="model",
        )
        # Upload config
        api.upload_file(
            path_or_fileobj=timm_config_path,
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
        )

        # Upload fgvc config
        api.upload_file(
            path_or_fileobj=fgvc_config_path,
            path_in_repo="config.yaml",
            repo_id=repo_name,
            repo_type="model",
        )
        
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
        )

    except Exception as exp:
        logging.warning(f"Error while uploading files to HuggingFace Hub:\n{exp}")

    return repo_name


def _create_timm_config(config: dict, config_path_json: str):
    """ Create a config file for timm. """
    timm_config = {
        "architecture": config["architecture"],
        "input_size": [3, *config["image_size"]],
        "num_classes": config["number_of_classes"]
    }
    with open(config_path_json, 'w') as fp:
        json.dump(timm_config, fp, indent=4)


def _create_model_repo_name(repo_owner: str, config: dict) -> str:
    """ Create a new HuggingFace model name """
    dataset = config["dataset"].lower()
    image_size = config["image_size"][-1]

    architecture = config["architecture"]
    architecture_split = architecture.split(".")
    if len(architecture_split) > 1:
        specification = architecture_split[1].split("_")[-1]
        definition = f"{architecture_split[0]}.{specification}_ft_{dataset}_{image_size}"
    else:
        definition = f"{architecture_split[0]}.ft_{dataset}_{image_size}"
    repo_name = f"{repo_owner}/{definition}"
    return repo_name


def get_default_model_card(config: dict, repo_name: str) -> str:
    """ Create a default model card for the DanishFungi project. """
    image_size = config["image_size"][-1]
    dataset = config["dataset"]

    model_mean = config.get("mean", "??")
    model_std = config.get("std", "??")

    model_card = \
f"""
---
tags:
- image-classification
- ecology
- fungi
- FGVC
library_name: DanishFungi
license: cc-by-nc-4.0
---
# Model card for {repo_name}

## Model Details
- **Model Type:** Danish Fungi Classification 
- **Model Stats:**
  - Params (M): ??
  - Image size: {image_size} x {image_size}
- **Papers:**
- **Original:** ??
- **Train Dataset:** {dataset} --> https://sites.google.com/view/danish-fungi-dataset

## Model Usage
### Image Embeddings
```python
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from urllib.request import urlopen
model = timm.create_model("hf-hub:{repo_name}", pretrained=True)
model = model.eval()
train_transforms = T.Compose([T.Resize(({image_size}, {image_size})), 
                              T.ToTensor(), 
                              T.Normalize({list(model_mean)}, {list(model_std)})]) 
img = Image.open(PATH_TO_YOUR_IMAGE)
output = model(train_transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
# output is a (1, num_features) shaped tensor
```

## Citation"""

    citations = """ 
```bibtex
@InProceedings{Picek_2022_WACV,
    author    = {Picek, Luk\'a\v{s} and \v{S}ulc, Milan and Matas, Ji\v{r}{\'\i} and Jeppesen, Thomas S. and Heilmann-Clausen, Jacob and L{\ae}ss{\o}e, Thomas and Fr{\o}slev, Tobias},
    title     = {Danish Fungi 2020 - Not Just Another Image Recognition Dataset},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1525-1535}
}
```
```bibtex
@article{picek2022automatic,
  title={Automatic Fungi Recognition: Deep Learning Meets Mycology},
  author={Picek, Luk{\'a}{\v{s}} and {\v{S}}ulc, Milan and Matas, Ji{\v{r}}{\'\i} and Heilmann-Clausen, Jacob and Jeppesen, Thomas S and Lind, Emil},
  journal={Sensors},
  volume={22},
  number={2},
  pages={633},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
"""
    model_card += citations
    return model_card


def create_model_card_file(model_card: str, exp_path: str) -> str:
    """ Create a model card file in the exp_path directory. Returns the path. """
    model_card_path = osp.join(exp_path, "README.md")
    with open(model_card_path, "w") as fp:
        fp.write(model_card)
    
    return model_card_path


def hfhub_load_args() -> tuple[argparse.Namespace, list[str]]:
    """Load script arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-path",
        help="Path to a exp directory with a valid config.yaml file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--repo-owner",
        help="Name of the HuggingFace repository owner (shortcut).",
        type=str,
        default=True,
    )
    parser.add_argument(
        "--saved-model",
        help="Specify to select a specific model to export (accuracy, f1, loss, "
        "recall, last_epoch).",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-card",
        help="Contents of the model card file.",
        type=str,
        required=False,
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def export_to_hfhub(
        *,
        exp_path: str = None,
        repo_owner: str = None,
        saved_model: str = None,
        model_card: str = None
) -> str:
    """ Wraps the export_to_huggingface_hub_from_checkpoint() with a CLI interface.
    Can be run from CLI with 'python hfhub.py --exp-path <exp_path> --repo-owner <repo_owner>
    (optionally --saved-model <saved_model> --model-card <model_card>)'
    '"""
    if exp_path is None or repo_owner is None:
        args, extra_args = hfhub_load_args()
        config = {"exp_path": args.exp_path}
        repo_owner = args.repo_owner
        saved_model = args.saved_model
        model_card = args.model_card
    else:
        config = {"exp_path": exp_path}

    return export_model_to_huggingface_hub_from_checkpoint(
        config=config,
        repo_owner=repo_owner,
        saved_model=saved_model,
        model_card=model_card
    )


if __name__ == "__main__":
    export_to_hfhub()
