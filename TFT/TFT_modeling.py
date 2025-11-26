import pandas as pd
import numpy as np

import torch
from torchmetrics import Metric

import lightning.pytorch as pl
from pytorch_forecasting import Baseline
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer

from sklearn.metrics import mean_absolute_error


class GroupFairnessMAE(Metric):
    # indicates that we do not need to recompute the whole state in each update call
    full_state_update = False

    def __init__(self, groups, **kwargs):
        super().__init__(**kwargs)

        # accumulate absolute errors across batches; use "cat" so distributed
        # reduction concatenates the lists from different processes
        self.add_state("abs_errors", default=[], dist_reduce_fx="cat")
        # accumulate group labels corresponding to the errors
        self.groups = groups

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        abs_error = (y_pred - y_true).abs()

        # detach and move to CPU to avoid retaining computation graph and
        # to make concatenation and reduction cheap and safe across devices
        self.abs_errors.append(abs_error.detach().cpu())

    def compute(self):
        # concatenate all stored tensors across batches (and across processes
        # in distributed settings because dist_reduce_fx="cat" was set)
        abs_errors = torch.cat(self.abs_errors)
        if self.groups.ndim > 1:
            group_vars = []
            n_groups = len(self.groups[0])
            for i in range(n_groups):
                groups = self.groups[:, i]
                group_vars.append(self.calculate_group_mae(abs_errors, groups))
            fairness_variance = torch.stack(group_vars).mean()
        else:
            fairness_variance = self.calculate_group_mae(abs_errors, self.groups)

        return fairness_variance

    def calculate_group_mae(self, abs_errors, groups):
        group_mae_list = []
        unique_groups = torch.unique(groups)
        for group in unique_groups:
            group_mask = groups == group
            group_errors = abs_errors[group_mask]
            group_mae = group_errors.mean()
            group_mae_list.append(group_mae)

        return torch.stack(group_mae_list).var()


def baseline_prediction(dataloader, metric):
    baseline_predictions = Baseline().predict(dataloader, return_y=True)
    return metric(baseline_predictions.output, baseline_predictions.y[0])


def train_model(
    tft, trainer, train_dataloader, val_dataloader, training, metric, logging_metrics
):
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")

    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=5,
        log_every_n_steps=1,
        accelerator="cpu",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=res.suggestion(),
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=metric,
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="adam",
        reduce_on_plateau_patience=4,
        logging_metrics=logging_metrics,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer
