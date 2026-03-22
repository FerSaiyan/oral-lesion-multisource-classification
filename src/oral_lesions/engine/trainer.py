# src/oral_lesions/engine/trainer.py
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .callbacks import EarlyStopping, CheckpointSaver

# Optional dependency: warmup scheduler is provided by legacy src.schedulers
try:
    from .schedulers import GradualWarmupScheduler  # not present in this package
except Exception:
    try:
        from ...schedulers import GradualWarmupScheduler  # fallback to src/schedulers.py
    except Exception:
        GradualWarmupScheduler = None  # not used directly here


class Trainer:
    def __init__(self, device, criterion, optimizer, scheduler=None, scaler=None, callbacks=()):
        self.device, self.criterion = device, criterion
        self.optimizer, self.scheduler = optimizer, scheduler
        self.scaler = scaler or torch.amp.GradScaler(enabled=False)
        self.callbacks = list(callbacks)

    @staticmethod
    def _unpack_batch(batch):
        """
        Supports:
          - (x, y)
          - (x, y, w)
          - (x, y, w, meta)
        Returns (x, y, w) where w may be None.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                x, y, w = batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                x, y = batch
                w = None
            else:
                x, y, w = batch, None, None
        else:
            x, y, w = batch, None, None
        return x, y, w

    @staticmethod
    def _reduce_loss(loss_vec, weight_tensor=None):
        """Handle vector or scalar loss and optional per-sample weights."""
        if loss_vec.dim() > 0:
            if weight_tensor is not None:
                return (loss_vec * weight_tensor).mean()
            return loss_vec.mean()
        return loss_vec

    def fit(self, model, train_loader, val_loader, epochs, log_csv_path, ckpt_path):
        best_f1 = 0.0
        history = []

        for epoch in range(epochs):
            # --- train ---
            model.train()
            tr_loss_sum = 0.0
            n_train_samples = 0

            for batch in train_loader:
                x, y, w = self._unpack_batch(batch)
                x = x.to(self.device, non_blocking=True)
                if y is not None:
                    y = y.to(self.device, non_blocking=True)
                if w is not None:
                    # Ensure per-sample weights stay in float32 on device
                    w = w.to(self.device, dtype=torch.float32, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type != "cpu")):
                    logits = model(x)
                    loss_vec = self.criterion(logits, y)
                    loss = self._reduce_loss(loss_vec, w)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                bs = x.size(0)
                tr_loss_sum += float(loss.item()) * bs
                n_train_samples += bs

            tr_loss = tr_loss_sum / max(1, n_train_samples)

            # --- validate ---
            model.eval()
            va_loss_sum, n_val_samples = 0.0, 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x, y, w = self._unpack_batch(batch)
                    x = x.to(self.device, non_blocking=True)
                    if y is not None:
                        y = y.to(self.device, non_blocking=True)
                    if w is not None:
                        w = w.to(self.device, dtype=torch.float32, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type != "cpu")):
                        logits = model(x)
                        l_vec = self.criterion(logits, y)
                        l = self._reduce_loss(l_vec, w)

                    bs = x.size(0)
                    va_loss_sum += float(l.item()) * bs
                    n_val_samples += bs
                    if y is not None:
                        y_true.extend(y.cpu().numpy())
                        y_pred.extend(torch.argmax(logits, 1).cpu().numpy())

            va_loss = va_loss_sum / max(1, n_val_samples)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
            acc = (np.array(y_true) == np.array(y_pred)).mean() * 100 if y_true else 0.0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{epochs} | LR {lr:.2e} | Train {tr_loss:.4f} | Val {va_loss:.4f} | F1 {f1:.4f} | Acc {acc:.2f}%")

            if self.scheduler:
                self.scheduler.step()

            history.append(
                dict(epoch=epoch + 1, train_loss=tr_loss, val_loss=va_loss, val_f1=f1, val_acc=acc, lr=lr)
            )
            pd.DataFrame(history).to_csv(log_csv_path, index=False)

            for cb in self.callbacks:
                cb.update(model, f1, ckpt_path)

            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    def fit_mix(
        self,
        model,
        train_loader_sup,
        val_loader,
        epochs,
        log_csv_path,
        ckpt_path,
        train_loader_pl=None,
        lambda_ramp_epochs: int = 20,
        test_loader=None,
        on_epoch_start=None,
        on_epoch_end=None,
    ):
        """
        Semi-supervised training: mixes supervised and optional pseudo-labeled batches.
        - lambda_u ramps linearly to 1.0 over lambda_ramp_epochs.
        - on_epoch_end(epoch, log_dict) if provided, is called after each epoch (for Optuna reporting/pruning).
        """
        best_f1, history = 0.0, []
        has_pl = train_loader_pl is not None

        for epoch in range(epochs):
            if on_epoch_start is not None:
                on_epoch_start(epoch, model)

            # --- train ---
            model.train()
            tr_loss_sum = 0.0

            lam = 0.0 if not has_pl else min(1.0, float(epoch + 1) / float(max(1, lambda_ramp_epochs)))
            it_sup = iter(train_loader_sup)
            it_pl = iter(train_loader_pl) if train_loader_pl is not None else None
            steps = max(len(train_loader_sup), len(train_loader_pl) if train_loader_pl is not None else 0)

            for _ in range(steps):
                sb = next(it_sup, None)
                pb = next(it_pl, None) if it_pl is not None else None
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type != "cpu")):
                    total = 0.0

                    # Supervised batch (may include sample weights)
                    if sb is not None:
                        xs, ys, ws = self._unpack_batch(sb)
                        xs = xs.to(self.device, non_blocking=True)
                        if ys is not None:
                            ys = ys.to(self.device, non_blocking=True)
                        if ws is not None:
                            ws = ws.to(self.device, dtype=torch.float32, non_blocking=True)

                        sup_logits = model(xs)
                        sup_loss_vec = self.criterion(sup_logits, ys)
                        sup_loss = self._reduce_loss(sup_loss_vec, ws)
                        total = total + sup_loss

                    # Pseudo-labeled batch (no sample weights)
                    if pb is not None:
                        xp, yp, _ = self._unpack_batch(pb)
                        xp = xp.to(self.device, non_blocking=True)
                        if yp is not None:
                            yp = yp.to(self.device, non_blocking=True)

                        pl_logits = model(xp)
                        pl_loss_vec = self.criterion(pl_logits, yp)
                        pl_loss = self._reduce_loss(pl_loss_vec, None)
                        total = total + lam * pl_loss

                if self.scaler.is_enabled():
                    self.scaler.scale(total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total.backward()
                    self.optimizer.step()

                bs_s = 0 if sb is None else self._unpack_batch(sb)[0].size(0)
                bs_p = 0 if pb is None else self._unpack_batch(pb)[0].size(0)
                tr_loss_sum += float(total.item()) * (bs_s + bs_p)

            denom = len(train_loader_sup.dataset) + (0 if train_loader_pl is None else len(train_loader_pl.dataset))
            tr_loss = tr_loss_sum / max(1, denom)

            # --- validate (val split) ---
            model.eval()
            va_loss_sum, n_val_samples = 0.0, 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x, y, w = self._unpack_batch(batch)
                    x = x.to(self.device, non_blocking=True)
                    if y is not None:
                        y = y.to(self.device, non_blocking=True)
                    if w is not None:
                        w = w.to(self.device, dtype=torch.float32, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type != "cpu")):
                        logits = model(x)
                        l_vec = self.criterion(logits, y)
                        l = self._reduce_loss(l_vec, w)

                    bs = x.size(0)
                    va_loss_sum += float(l.item()) * bs
                    n_val_samples += bs
                    if y is not None:
                        y_true.extend(y.cpu().numpy())
                        y_pred.extend(torch.argmax(logits, 1).cpu().numpy())

            va_loss = va_loss_sum / max(1, n_val_samples)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) if y_true else 0.0
            acc = (np.array(y_true) == np.array(y_pred)).mean() * 100 if y_true else 0.0
            lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler:
                self.scheduler.step()

            # --- optional test evaluation ---
            test_loss = None
            f1_macro_test = None
            f1_weighted_test = None
            acc_test = None
            if test_loader is not None:
                te_loss_sum, n_te_samples = 0.0, 0
                y_true_te, y_pred_te = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        x, y, w = self._unpack_batch(batch)
                        x = x.to(self.device, non_blocking=True)
                        if y is not None:
                            y = y.to(self.device, non_blocking=True)
                        if w is not None:
                            w = w.to(self.device, dtype=torch.float32, non_blocking=True)

                        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type != "cpu")):
                            logits = model(x)
                            l_vec = self.criterion(logits, y)
                            l = self._reduce_loss(l_vec, w)

                        bs = x.size(0)
                        te_loss_sum += float(l.item()) * bs
                        n_te_samples += bs
                        if y is not None:
                            y_true_te.extend(y.cpu().numpy())
                            y_pred_te.extend(torch.argmax(logits, 1).cpu().numpy())

                test_loss = te_loss_sum / max(1, n_te_samples)
                f1_macro_test = (
                    f1_score(y_true_te, y_pred_te, average="macro", zero_division=0) if y_true_te else 0.0
                )
                f1_weighted_test = (
                    f1_score(y_true_te, y_pred_te, average="weighted", zero_division=0) if y_true_te else 0.0
                )
                acc_test = (np.array(y_true_te) == np.array(y_pred_te)).mean() * 100 if y_true_te else 0.0

            # Logging message including val + (optionally) test metrics
            base_msg = (
                f"Epoch {epoch+1}/{epochs} | LR {lr:.2e} | "
                f"Train {tr_loss:.4f} | Val {va_loss:.4f} | "
                f"ValF1_macro {f1_macro:.4f} | ValF1_weighted {f1_weighted:.4f} | ValAcc {acc:.2f}%"
            )
            if test_loss is not None:
                base_msg += (
                    f" | Test {test_loss:.4f} | "
                    f"TestF1_macro {f1_macro_test:.4f} | "
                    f"TestF1_weighted {f1_weighted_test:.4f} | TestAcc {acc_test:.2f}%"
                )
            if has_pl:
                base_msg += f" | lam {lam:.2f}"
            print(base_msg)

            # Per-epoch log row (val metrics always recorded; test metrics optional)
            row = dict(
                epoch=epoch + 1,
                train_loss=tr_loss,
                val_loss=va_loss,
                val_f1=f1_macro,
                val_f1_weighted=f1_weighted,
                val_acc=acc,
                lr=lr,
            )
            if test_loss is not None:
                row.update(
                    test_loss=test_loss,
                    test_f1=f1_macro_test,
                    test_f1_weighted=f1_weighted_test,
                    test_acc=acc_test,
                )
            if has_pl:
                row["lambda_u"] = lam

            history.append(row)
            pd.DataFrame(history).to_csv(log_csv_path, index=False)

            for cb in self.callbacks:
                cb.update(model, f1_macro, ckpt_path)

            if on_epoch_end is not None:
                on_epoch_end(epoch, row)

            if f1_macro > best_f1:
                best_f1 = f1_macro

        return best_f1
