"""
This module provides functionality to extract image embeddings using a specified
pretrained model from the torchvision library. It includes functions to:
- List image files directly from a ZIP file without extraction.
- Apply model-specific preprocessing and transformations.
- Extract embeddings using various models.
- Save the resulting embeddings into a CSV file.
Modules required:
- argparse: For command-line argument parsing.
- os, csv, zipfile: For file handling (ZIP file reading, CSV writing).
- inspect: For inspecting function signatures and models.
- torch, torchvision: For loading and using pretrained models to extract embeddings.
- PIL, cv2: For image processing tasks such as resizing, normalization, and conversion.
"""

import argparse
import csv
import inspect
import logging
import os
import zipfile
from inspect import signature

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, get_worker_info

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Available models from torchvision
AVAILABLE_MODELS = {
    name: getattr(models, name)
    for name in dir(models)
    if callable(getattr(models, name)) and "weights" in signature(getattr(models, name)).parameters
}

# Default resize and normalization settings for models
MODEL_DEFAULTS = {
    "default": {"resize": (224, 224), "normalize": (
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )},
    "efficientnet_b1": {"resize": (240, 240)},
    "efficientnet_b2": {"resize": (260, 260)},
    "efficientnet_b3": {"resize": (300, 300)},
    "efficientnet_b4": {"resize": (380, 380)},
    "efficientnet_b5": {"resize": (456, 456)},
    "efficientnet_b6": {"resize": (528, 528)},
    "efficientnet_b7": {"resize": (600, 600)},
    "inception_v3": {"resize": (299, 299)},
    "swin_b": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "swin_s": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "swin_t": {"resize": (224, 224), "normalize": (
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]
    )},
    "vit_b_16": {"resize": (224, 224), "normalize": (
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    )},
    "vit_b_32": {"resize": (224, 224), "normalize": (
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    )},
}
for m, settings in MODEL_DEFAULTS.items():
    if "normalize" not in settings:
        settings["normalize"] = MODEL_DEFAULTS["default"]["normalize"]

# Custom transform classes
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        img = np.array(img.convert("L"))
        img = self.clahe.apply(img)
        return Image.fromarray(img).convert("RGB")


class CannyTransform:
    def __init__(self, threshold1=100, threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, img):
        img = np.array(img.convert("L"))
        edges = cv2.Canny(img, self.threshold1, self.threshold2)
        return Image.fromarray(edges).convert("RGB")


class RGBAtoRGBTransform:
    def __call__(self, img):
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")
        return img


def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with its own ZipFile handle.
    """
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.zip_ref = zipfile.ZipFile(dataset.zip_file, "r")


class ImageDataset(Dataset):
    """
    Dataset for reading images directly from a ZIP file.
    """
    def __init__(self, zip_file, file_list, transform=None):
        self.zip_file = zip_file
        self.file_list = file_list
        self.transform = transform
        self.zip_ref = None  # set in worker_init_fn

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # use worker-local handle if present
        zf = self.zip_ref or zipfile.ZipFile(self.zip_file, "r")
        with zf.open(self.file_list[idx]) as fh:
            img = Image.open(fh)
            if self.transform:
                img = self.transform(img)
            return img, os.path.basename(self.file_list[idx])


def collate_fn(batch):
    """
    Collate that filters out invalid items and batches tensors.
    """
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, names = zip(*batch)
    return torch.stack(images), names


def get_image_files_from_zip(zip_file):
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            return [f for f in zip_ref.namelist() if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".gif")
            )]
    except zipfile.BadZipFile:
        raise RuntimeError(f"Invalid ZIP file: {zip_file}")


def load_model(model_name, device):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    try:
        model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)
        logging.info(f"Model {model_name} loaded")
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        raise
    # strip classifier layers
    if hasattr(model, "fc"): model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"): model.classifier = torch.nn.Identity()
    elif hasattr(model, "head"): model.head = torch.nn.Identity()
    model.eval()
    return model


def write_csv(output_csv, list_embeddings, ludwig_format=False):
    with open(output_csv, mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if list_embeddings:
            if ludwig_format:
                writer.writerow(["sample_name", "embedding"]);
                for emb in list_embeddings:
                    name, vec = emb[0], emb[1:]
                    writer.writerow([name, " ".join(map(str, vec))])
            else:
                header = ["sample_name"] + [f"vector{i+1}" for i in range(len(list_embeddings[0])-1)]
                writer.writerow(header)
                writer.writerows(list_embeddings)
        else:
            writer.writerow(["sample_name", "embedding"] if ludwig_format else ["sample_name"])


def extract_embeddings(
        model_name, apply_normalization,
        zip_file, file_list, transform_type="rgb"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)
    settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = settings["resize"]
    normalize = settings["normalize"]

    # select initial transform
    if transform_type == "grayscale":
        init = transforms.Grayscale(num_output_channels=3)
    elif transform_type == "clahe":
        init = CLAHETransform()
    elif transform_type == "edges":
        init = CannyTransform()
    elif transform_type == "rgba_to_rgb":
        init = RGBAtoRGBTransform()
    else:
        init = transforms.Lambda(lambda x: x.convert("RGB"))

    pipeline = [init, transforms.Resize(resize), transforms.ToTensor()]
    if apply_normalization:
        pipeline.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
    transform = transforms.Compose(pipeline)

    list_embeddings = []
    with torch.inference_mode():
        # parallel loader
        dataset = ImageDataset(zip_file, file_list, transform)
        loader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=(device=="cuda"),
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn
        )
        try:
            for imgs, names in loader:
                if imgs is None: continue
                imgs = imgs.to(device)
                embs = model(imgs).cpu().numpy()
                for nm, e in zip(names, embs):
                    list_embeddings.append([nm] + e.tolist())
        except (RuntimeError, zipfile.BadZipFile) as e:
            logging.warning(f"Parallel load failed ({type(e).__name__}: {e}), falling back.")
            # sequential fallback
            for f in file_list:
                try:
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        with zf.open(f) as fh:
                            img = Image.open(fh)
                            img = transform(img)
                            inp = img.unsqueeze(0).to(device)
                            emb = model(inp).squeeze().cpu().numpy()
                            list_embeddings.append([os.path.basename(f)] + emb.tolist())
                except Exception as se:
                    logging.warning(f"Skipping {f}: {se}")

    return list_embeddings


def main(zip_file, output_csv, model_name,
         apply_normalization=False, transform_type="rgb", ludwig_format=False):
    files = get_image_files_from_zip(zip_file)
    embs = extract_embeddings(model_name, apply_normalization, zip_file, files, transform_type)
    write_csv(output_csv, embs, ludwig_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings.")
    parser.add_argument("--zip_file", required=True, help="ZIP with images.")
    parser.add_argument("--model_name", required=True, choices=AVAILABLE_MODELS.keys())
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--transform_type", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--ludwig_format", action="store_true")
    args = parser.parse_args()
    main(
        args.zip_file,
        args.output_csv,
        args.model_name,
        args.normalize,
        args.transform_type,
        args.ludwig_format
    )
