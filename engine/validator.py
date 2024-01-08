import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

# from ultralytics.nn.autobackend import AutoBackend
# from ultralytics.yolo.cfg import get_cfg
# from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
# from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.files import increment_path
# from ultralytics.yolo.utils.ops import Profile
# from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from utils.cfg_utils import get_cfg
from utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM_BAR_FORMAT,
    callbacks,
    colorstr,
    emojis,
    Profile,
)
from utils.flags import Mode
from utils.checks import check_imgsz
from utils.torch_utils import de_parallel, select_device


class BaseValidator:
    """
    BaseValidator

    A base class for creating validators.

    Attributes:
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        args (SimpleNamespace): Configuration for the validator.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        speed (float): Batch processing speed in seconds.
        jdict (dict): Dictionary to store validation results.
        save_dir (Path): Directory to save results.
    """

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
        """
        self.dataloader = dataloader
        if dataloader is not None:
            self.testset = dataloader.dataset
        else:
            self.testset = None

        self.pbar = pbar
        self.args = args or get_cfg(DEFAULT_CFG)
        self.model = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.jdict = None
        self.pose_evaluator = None

        # project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        #name = self.args.name or f"{self.args.mode}"
        if not save_dir and args is not None:
            self.save_dir = Path(self.args.save_dir) / "val"
            (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(
                parents=True, exist_ok=True
            )
        else:
            self.save_dir = Path(save_dir)
        self.csv_path = self.save_dir / "bop_results.csv"

        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @torch.no_grad()
    def __call__(self, trainer=None, model=None, transforms=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            # self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            #self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            #self.transforms = trainer.transforms
            self.model = model
            self.loss_names = trainer.loss_names
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.save_csv = False
            #self.criterion = trainer.criterion
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_val_start")
            assert model is not None, "Either trainer or model is needed for validation"
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != "cpu"
            #model = model 
            # make it possible to load from string, trt, etc.
            self.model = model.float().to(self.device).eval()
            self.transforms = transforms

            if self.device.type == "cpu":
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            self.dataloader = self.dataloader or self.get_dataloader(
                mode=Mode.TEST
            )
            self.testset = self.dataloader.dataset
            self.init_metrics()

            if self.args.save_csv:
                self.init_csv()

            model.eval()
#            model.warmup(
#                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
#            )  # warmup
#
        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics() #de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch)

            # Postprocess
            with dt[2]:
                preds, batch = self.postprocess(preds, batch=batch)

            # Loss
            #with dt[3]:
            #    if self.training:
            #        self.loss += self.criterion(preds, batch)[1]
            if self.args.save_csv:
                self.save_csv(preds, batch, dt)
            self.update_metrics(preds, batch)
            self.run_callbacks("on_val_batch_end")

        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.seen = 0
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {
                **self.metrics.results_dict,
                **trainer.label_loss_items(
                    self.loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            #if self.args.save_json and self.pose_evaluator:
            #    self.pose_evaluator.write_all(self.save_dir)
            return self.metrics.results_dict #stats
    
    def criterion(self, batch, preds):
        raise NotImplementedError("criterion function not implemented in validator")

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_pose_evaluator(self):
        """Returns the pose evaluator."""
        raise NotImplementedError(
            "get_pose_evaluator function not implemented in validator"
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError(
            "get_dataloader function not implemented for this validator"
        )

    def build_dataset(self, img_path):
        """Build dataset"""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in training/validation."""
        return []

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots model predictions on batch images."""
        pass
    
    def save_csv(self, preds, batch, dt):
        """Saves predictions in bop format to csv file."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {'data': data, 'timestamp': time.time()}
    
    def init_csv(self):
        with open(self.csv_path, "w") as f:
            f.write("scene_id,im_id,obj_id,score,R,t,time\n")