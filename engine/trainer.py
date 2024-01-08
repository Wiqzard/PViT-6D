import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from utils import RANK,ROOT, LOGGER,TQDM_BAR_FORMAT, DEFAULT_CFG, yaml_save, colorstr,  increment_dir
from utils.cfg_utils import get_cfg, print_args
from utils.torch_utils import select_device, ModelEMA, de_parallel, EarlyStopping, generate_ddp_command, ddp_cleanup, one_cycle, strip_optimizer, attempt_load_one_weight
from utils.checks import check_imgsz, check_file
from utils.flags import Mode 

from utils import callbacks

class BaseTrainer:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        
        self.i = 0
        

        # Dirs
        project = self.args.project
        name = self.args.name 
        if hasattr(self.args, "save_dir") and self.args.save_dir:
            self.save_dir = Path(self.args.save_dir) / project / name
        else:
            self.save_dir = ROOT / "runs" / str(project) / name  
        self.save_dir = increment_dir(self.save_dir)  # increment run
        
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = (
            self.wdir / "last.pt",
            self.wdir / "best.pt",
        )  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == "cpu":
            self.args.workers = (
                0  # faster CPU training as time dominated by inference, not dataloading
            )

        # Model and Dataset
        self.model = self.args.model
        self.dataset_path = self.args.dataset_path
        self.trainset, self.testset = None ,None 
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [1, 2000, 10000]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if (
            isinstance(self.args.device, int) or self.args.device
        ):  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"Running DDP command {cmd}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        LOGGER.info(
            f"DDP settings: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}"
        )
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=3600),
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model.to(self.device)
        self.set_model_attributes()
#        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = (
                callbacks.default_callbacks.copy()
            )  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(self.args.amp, device=self.device)
#            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1:  # DDP
            dist.broadcast(
                self.amp, src=0
            )  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = self.args.amp
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK]) #, find_unused_parameters=True)
        # Check imgsz
        gs = max(
            int(self.model.stride.max() if hasattr(self.model, "stride") else 16), 16
        )  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Optimizer
        self.accumulate = max(
            round(self.args.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        weight_decay = (
            self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        )  # scale weight_decay
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
        )
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf
            )  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # Dataloaders
        batch_size = (
            self.batch_size // world_size if world_size > 1 else self.batch_size
        )
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=RANK, mode=Mode.TRAIN
        )
        if RANK in (-1, 0):
            if self.args.val:
                self.test_loader = self.get_dataloader(
                    self.testset, batch_size=batch_size * 2, rank=-1, mode=Mode.TEST
                )
                self.validator = self.get_validator()
                metric_keys = self.validator.metrics.keys + self.label_loss_items(
                    prefix="val"
                )
                self.metrics = dict(
                    zip(metric_keys, [0] * len(metric_keys))
                )  # TODO: init metrics for plot_results()?
            else:
                self.metrics = {}
            self.ema = ModelEMA(self.model)
         #   if self.args.plots and not self.args.v5loader:
         #       self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(
            round(self.args.warmup_epochs * nb), 100
        )  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs} epochs..."
        )
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(
                    enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT
                )
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1,
                        np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round(),
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch)
                    preds, batch = self.postprocess_batch(preds, batch)
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None
                        else self.loss_items
                    )
                    #self.tloss = self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                lr = self.optimizer.param_groups[0]["lr"]
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.5g" + "%11.4g" * ( loss_len)) #+2
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            mem,
                            round(lr, 6),
                            *losses,
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch,preds, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers

            self.scheduler.step()
            self.run_callbacks("on_train_epoch_end")

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    if self.args.val_period > 0 and (
                        (epoch + 1) % self.args.val_period == 0 or final_epoch
                    ):
                        self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks("on_model_save")

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            if self.args.final_val:
                LOGGER.info(f"Starting final validation on {self.args.data}...")
                self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model checkpoints based on various conditions."""
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": de_parallel(self.model), #.half(),
            #"model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "transforms": self.transforms,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "date": datetime.now().isoformat(),
        }

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (
            (self.epoch > 0)
            and (self.save_period > 0)
            and (self.epoch % self.save_period == 0)
        ):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")
        del ckpt

    @staticmethod
    def get_dataset(path):
        raise NotImplementedError
    
    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(
            self.model, torch.nn.Module
        ):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(model)
            #cfg = ckpt["model"].yaml
        else:
            cfg = self.args #model
        self.model = self.get_model(
             weights=weights, verbose=RANK == -1
        ) 
        return ckpt



    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=0.5 #1.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def postprocess_batch(self, pred, batch):
        """
        Allows custom postprocessing of model outputs and ground truths depending on task type.
        """
        pass

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop(
            "fitness", -self.loss.detach().cpu().numpy()
        )  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")
        self.loss_names = ["loss"]
        return 

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError("get_dataloader function not implemented in trainer")


    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        raise NotImplementedError("criterion function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        #return {'loss': loss_items} if loss_items is not None else ['loss']
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        #self.model.names = self.args.names
        self.model.cfg = self.args

    def build_targets(self, preds, targets):
        """Builds target tensors for training model model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, preds, ni):
        """Plots training samples during modelv5 training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for model model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = (
            ""
            if self.csv.exists()
            else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")
        )  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection model model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    f = self.get_model(weights=f)
                    LOGGER.info(f"\nValidating {f}...")
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                resume = self.args.model
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume))

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                _ , ckpt = attempt_load_one_weight(last)#.args
                ckpt_args = ckpt["train_args"] #model.args
                self.args = get_cfg(ckpt_args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from "
                    "i.e. 'train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        if self.resume:
            assert start_epoch > 0, (
                f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
                f"Start a new training without resuming'"
            )
            LOGGER.info(
                f"Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs"
            )
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch

    @staticmethod
    def build_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, "bias") and isinstance(
                v.bias, nn.Parameter
            ):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(
                v.weight, nn.Parameter
            ):  # weight (with decay)
                g[0].append(v.weight)

        if name == "Adam":
            optimizer = torch.optim.Adam(
                g[2], lr=lr, betas=(momentum, 0.999)
            )  # adjust beta1 to momentum
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer {name} not implemented.")

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
        )
        return optimizer

