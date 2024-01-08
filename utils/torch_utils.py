import math
import time
import os
import platform
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Union
import re
import shutil
import socket
import sys
import tempfile


import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, distributed
import torchvision


from . import USER_CONFIG_DIR

try:
    import thop
except ImportError:
    thop = None
from utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER,  RANK, VERSION , PIN_MEMORY, emojis, check_version
from utils.checks import check_suffix
TORCHVISION_0_10 = check_version(torchvision.__version__, '0.10.0')
TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_1_11 = check_version(torch.__version__, '1.11.0')
TORCH_1_12 = check_version(torch.__version__, '1.12.0')
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')#


def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = f'Mesh Pose {VERSION} 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'

    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f'{trainer.__class__.__module__}.{trainer.__class__.__name__}'.rsplit('.', 1)

    content = f'''overrides = {vars(trainer.args)} \nif __name__ == "__main__":
    from {module} import {name}

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.train()'''
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_',
                                     suffix=f'{id(trainer)}.py',
                                     mode='w+',
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = str(Path(sys.argv[0]).resolve())
    safe_pattern = re.compile(r'^[a-zA-Z0-9_. /\\-]{1,128}$')  # allowed characters and maximum of 100 characters
    if not (safe_pattern.match(file) and Path(file).exists() and file.endswith('.py')):  # using CLI
        file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f'{id(trainer)}.py' in file:  # if temp_file suffix in file
        os.remove(file)
        
        
def get_num_params(model):
    """Return the total number of parameters in a model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640):
    """Return a model's FLOPs."""
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1E9 * 2 if thop else 0  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        return flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        return 0

def model_info_for_loggers(trainer):
    """
    Return model info dict with useful model information.

    Example for YOLOv8n:
        {'model/parameters': 3151904,
         'model/GFLOPs': 8.746,
         'model/speed_ONNX(ms)': 41.244,
         'model/speed_TensorRT(ms)': 3.211,
         'model/speed_PyTorch(ms)': 18.755}
    """
    #if trainer.args.profile:  # profile ONNX and TensorRT times
        #from ultralytics.yolo.utils.benchmarks import ProfileModels
        #results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        #results.pop('model/name')
    #else:  # only return PyTorch times from most recent validation
    results = {
            'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3)}
    results['model/speed_PyTorch(ms)'] = round(trainer.validator.speed['inference'], 3)
    return results

def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)




class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.

    Returns:
        None

    Usage:
        from pathlib import Path
        from ultralytics.yolo.utils.torch_utils import strip_optimizer
        for f in Path('/Users/glennjocher/Downloads/weights').rglob('*.pt'):
            strip_optimizer(f)
    """
    # Use dill (if exists) to serialize the lambda functions where pickle does not do this
    try:
        import dill as pickle
    except ImportError:
        import pickle

    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**x['train_args']} if 'train_args' in x else None  # combine args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'best_fitness', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f, pickle_module=pickle)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


class EarlyStopping:
    """
    Early stopping class that stops training when a specified number of epochs have passed without improvement.
    """

    def __init__(self, patience=50):
        """
        Initialize early stopping object

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `patience=300` or use `patience=0` to disable EarlyStopping.')
        return stop

        
        
#def torch_safe_load(weight):
#    """
#    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
#    it catches the error, logs a warning message, and attempts to install the missing module via the
#    check_requirements() function. After installation, the function again attempts to load the model using torch.load().
#
#    Args:
#        weight (str): The file path of the PyTorch model.
#
#    Returns:
#        (dict): The loaded PyTorch model.
#    """
#
#    check_suffix(file=weight, suffix='.pt')
#    file = weight  # search online if missing locally
#    try:
#        return torch.load(file, map_location='cpu'), file  # load
#    except ModuleNotFoundError as e:  # e.name is missing module name
#        if e.name == 'models':
#            raise TypeError(
#                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
#                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
#                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
#        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
#                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
#                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
#                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
#        check_requirements(e.name)  # install missing module
#
#        return torch.load(file, map_location='cpu'), file  # load
#

 
def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch.load(weight, map_location="cpu"), weight  # load
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([16.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])

        
class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers. Uses same syntax as vanilla DataLoader."""

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        self.time = 0
        for _ in range(len(self)):
            start_time = time.time()
            yield next(self.iterator)
            self.time = time.time() - start_time

    def reset(self):
        """Reset iterator.
        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()

        
class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)

def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)

                              
def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

    
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        

    def load(self, weights, verbose=True):
        model = weights['model'] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.model.state_dict())
        self.model.load_state_dict(csd, strict=False)
        if verbose:
            print(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

def clear_cuda_memory() -> None:
    if TORCH_2_0:
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()