# utils/trainer.py
import os
import csv
import time
import math
from pathlib import Path

import torch
import numpy as np

# Optional metrics (macro P/R/F1). Falls back to acc-only if not available.
try:
    from utils.metrics import precision_recall_f1  # (logits, targets, average="macro", num_classes=None)
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False


class trainer:
    """
    Training utility with:
      - CUDA AMP (mixed precision)
      - Gradient accumulation
      - Gradient clipping
      - Resume (model/opt/sched/scaler)
      - Early stopping (by val loss)
      - Safer atomic checkpoints
      - CSV logging (loss/acc + optional P/R/F1)
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        val_loader,
        device,
        output_dir="outputs",
        resume_ckpt=None,
        amp=True,
        grad_clip_norm=None,
        accum_steps=1,
        early_stopping_patience=None,
        save_every=0,  # 0 => only save best + last
    ):
        self.model = model
        self.opt = optimizer
        self.criterion = criterion
        self.sched = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.amp = bool(amp) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.grad_clip_norm = grad_clip_norm
        self.accum_steps = max(1, int(accum_steps))
        self.early_stopping_patience = early_stopping_patience
        self.save_every = int(save_every)

        self.start_epoch = 0
        self.best_val_loss = math.inf
        self._no_improve_epochs = 0

        self.model.to(self.device)

        if resume_ckpt:
            self._load_checkpoint(resume_ckpt)

        # CSV Logger
        self.log_path = os.path.join(self.logs_dir, "metrics.csv")
        if not os.path.exists(self.log_path) or self.start_epoch == 0:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
                if HAS_METRICS:
                    header += ["val_precision", "val_recall", "val_f1"]
                writer.writerow(header)

    # --------------- public API ---------------
    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_loss = self._train_one_epoch()

            val_metrics = self._validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            lr = self._current_lr()

            # CSV log
            row = [epoch + 1, train_loss, val_loss, val_acc, lr]
            if HAS_METRICS:
                row += [val_metrics["precision"], val_metrics["recall"], val_metrics["f1"]]
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # Console log
            msg = (f"Epoch {epoch+1}/{epochs} | "
                   f"train {train_loss:.4f} | "
                   f"val {val_loss:.4f} acc {val_acc:.3f}")
            if HAS_METRICS:
                msg += f" p {val_metrics['precision']:.3f} r {val_metrics['recall']:.3f} f1 {val_metrics['f1']:.3f}"
            msg += f" | lr {lr:.6f} | {time.time() - t0:.1f}s"
            print(msg)

            # Checkpointing
            is_best = val_loss < self.best_val_loss
            self._save_checkpoint(epoch, is_best)

            # Early stopping
            if is_best:
                self.best_val_loss = val_loss
                self._no_improve_epochs = 0
            else:
                self._no_improve_epochs += 1
                if self.early_stopping_patience is not None and \
                   self._no_improve_epochs >= self.early_stopping_patience:
                    print(f"[EarlyStopping] No improvement for {self._no_improve_epochs} epochs. Stopping.")
                    break

            # Scheduler step (after val)
            if self.sched is not None:
                try:
                    self.sched.step(val_loss)  # supports ReduceLROnPlateau
                except TypeError:
                    self.sched.step()          # cosine/step schedulers

        # Save final (last) checkpoint
        self._save_checkpoint(epoch, is_best=False, last_only=True)

    # --------------- internal helpers ---------------
    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        self.opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(self.train_loader, start=1):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if step % self.accum_steps == 0:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size * self.accum_steps  # undo division
            num_samples += batch_size

        return running_loss / max(1, num_samples)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # For macro metrics (optional)
        macro_p = macro_r = macro_f1 = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            if HAS_METRICS:
                p, r, f1 = precision_recall_f1(logits, y, average="macro", num_classes=logits.shape[1])
                macro_p += p
                macro_r += r
                macro_f1 += f1
                n_batches += 1

        metrics = {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total),
        }
        if HAS_METRICS and n_batches > 0:
            metrics.update({
                "precision": macro_p / n_batches,
                "recall": macro_r / n_batches,
                "f1": macro_f1 / n_batches,
            })
        return metrics

    def _current_lr(self):
        if len(self.opt.param_groups) > 0:
            return float(self.opt.param_groups[0].get("lr", 0.0))
        try:
            return float(self.sched.get_last_lr()[0])
        except Exception:
            return 0.0

    def _save_checkpoint(self, epoch, is_best=False, last_only=False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
            "best_val_loss": self.best_val_loss,
            "amp": self.amp,
        }
        # Save scheduler & scaler state if available
        if self.sched is not None:
            try:
                state["sched_state"] = self.sched.state_dict()
            except Exception:
                pass
        try:
            state["scaler_state"] = self.scaler.state_dict()
        except Exception:
            pass

        def _atomic_save(path):
            tmp = f"{path}.tmp"
            torch.save(state, tmp)
            os.replace(tmp, path)

        # Save 'last'
        last_path = os.path.join(self.ckpt_dir, "last.pth")
        _atomic_save(last_path)

        # Save 'best'
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best.pth")
            _atomic_save(best_path)

        # Save epoch checkpoint if requested
        if not last_only and self.save_every > 0 and ((epoch + 1) % self.save_every == 0):
            ep_path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1:03d}.pth")
            _atomic_save(ep_path)

    def _load_checkpoint(self, path):
        print(f"[Resume] Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        if "opt_state" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt_state"])
            except Exception as e:
                print(f"[Resume] Warning: could not load optimizer state: {e}")
        if "sched_state" in ckpt and self.sched is not None:
            try:
                self.sched.load_state_dict(ckpt["sched_state"])
            except Exception as e:
                print(f"[Resume] Warning: could not load scheduler state: {e}")
        if "scaler_state" in ckpt and self.amp:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"[Resume] Warning: could not load scaler state: {e}")
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_val_loss", math.inf)
