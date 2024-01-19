from collections import OrderedDict
from inspect import signature
import os
import numpy as np
from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F

from ssl_fork.core.criterions.consistency import ConsistencyLoss
from ssl_fork.core.criterions.cross_entropy import CELoss


class AlgorithmBase:
    """
    Base class for algorithms
    init algorithm specific parameters and common parameters

    Args:

    """

    def __init__(self, args, tb_log=None, logger=None, **kwargs) -> None:
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.lambda_u = args.ulb_loss_ratio
        self.algorithm = args.algorithm

        # common utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info

        # common model-related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        self.loader_dict = self.set_data_loader()

        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        self.optimizer, self.scheduler = self.set_optimizer()

        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # set common hooks during training
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()
        self.set_hooks()

    def init(self, **kwargs):
        raise NotImplementedError

    def set_dataset(self):
        dataset_dict = {}
        return dataset_dict

    def process_batch(self, input_args=None, **kwargs):
        """send data to gpu"""
        input_dict = {}

        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        for arg, var in kwargs.items():
            var = var.cuda(self.gpu)
            input_dict[arg] = var

        return input_dict

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train step specific to each algorithm

        returns two dictionaries
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # record log_dict
        # return log_dict
        raise NotImplementedError

    def train(self):
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            for data_lb, data_ulb in zip(
                self.loader_dict["train_lb"], self.load_dict["train_ulb"]
            ):
                self.out_dict, self.log_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb)
                )
                self.it += 1

    def process_log_dict(self, log_dict=None, prefix="train", **kwargs):
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f"{prefix}/" + arg] = var
        return log_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """add to out dict"""
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var

        return out_dict

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        self.model.eval()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]

                # why is it ignoring the last index
                loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu.tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_logits = np.concatenate(y_logits)
            top1 = accuracy_score(y_true, y_pred)
            balanced_top1 = precision_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            F1 = f1_score(y_true, y_pred, average="macro")

            cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            self.print_fn("confusion matrix :\n" + np.array_str(cf_mat))

            self.model.train()

            eval_dict = {
                eval_dest + "/loss": total_loss / total_num,
                eval_dest + "/top-1-acc": top1,
                eval_dest + "/balanced_acc": balanced_top1,
                eval_dest + "/precision": precision,
                eval_dest + "/recall": recall,
                eval_dest + "/F1": F1,
            }
            if return_logits:
                eval_dict[eval_dest + "/logits"] = y_logits
            return eval_dict

    def get_save_dict(self):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "it": self.it + 1,
            "epoch": self.epoch + 1,
            "best_it": self.best_it,
            "best_eval_acc": self.best_eval_acc,
        }
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()

        return save_dict

    def save_model(self, save_name, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.it = checkpoint["it"]
        self.start_epoch = checkpoint["epoch"]
        self.epoch = self.start_epoch
        self.best_it = checkpoint["best_it"]
        self.best_eval_acc = checkpoint["best_eval_acc"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.print_fn("Model loaded")

        return checkpoint
