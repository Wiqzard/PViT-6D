import os
import warnings
from pathlib import Path
import argparse

import torch
from torchvision import transforms

from utils import IterableSimpleNamespace
from utils.cfg_utils import get_cfg
from utils.torch_utils import clear_cuda_memory
from exp.direct.direct_trainer import DirectTrainer
from exp.direct.direct_validator import DirectValidator


def train() -> int:
    clear_cuda_memory()
    os.environ["MASTER_ADDR"] = "localhost"
    torch.backends.cudnn.benchmark = True
    os.environ["OMP_NUM_THREADS"] = "32"
    cfg = get_cfg(
        "configs/direct_method.yaml"
    )
    trainer = DirectTrainer(cfg)
    trainer.train()
    return 0


def val() -> int:
    warnings.filterwarnings("ignore")
    clear_cuda_memory()

    #path = "runs/debug/mvitv2token_large_fcos_2/weights/best.pt"
    path = args.model 

    ckpt = torch.load(path, map_location="cpu")
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if "ema" in ckpt:
        model = ckpt.pop("ema")
        if model is None:
            model = ckpt.pop("model")
    else:
        model = ckpt.pop("model")

    model = model.float().eval()
    args = ckpt.pop("train_args")
    args = IterableSimpleNamespace(**args)
    args.aug = False
    #args.det_dataset = "fcos"  # None|detr|fcos|faster_rcnn|faster_rcnn_50
    args.save_dir = Path(path).parent.parent.__str__()

    save_dir = args.save_dir + "/eval/" + args.det_dataset  ##detr_95"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    validator = DirectValidator(save_dir=save_dir, args=args)
    results = validator(model=model, transforms=preprocess)
    return 0

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training and Validation Script")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val'], help='Mode of operation: train or val')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'val':
        val()

#if __name__ == "__main__":
#    main()


if __name__ == "__main__":
    train()
    #val()
