"""Training loop with validation, early stopping and history for the plots."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from .data import iterate_batches, prepare_batch


@dataclass
class TrainConfig:
    name: str
    image_size: int = 96
    batch_size: int = 128
    micro_batch: int = 256  # gradient accumulation chunk (memory only, no effect on the math)
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "Adam"
    augment: float = 0.0  # 0 = no augmentation, 1 = full strength
    patience: int = 5  # early stopping on validation loss
    label_smoothing: float = 0.0
    cosine_schedule: bool = False
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.25, 0.25, 0.25)
    seed: int = 42


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)
    best_epoch: int = 0
    total_time: float = 0.0


def make_optimizer(cfg: TrainConfig, params) -> torch.optim.Optimizer:
    if cfg.optimizer == "Adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "AdamW":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "SGD":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate(model, x, y, cfg: TrainConfig, device, batch_size: int = 256):
    """Returns (mean loss, accuracy, softmax probabilities on CPU)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, probs = 0.0, 0, []
    for idx in iterate_batches(len(y), batch_size, shuffle=False):
        xb = prepare_batch(x[idx], device, cfg.image_size, cfg.mean, cfg.std)
        yb = y[idx].to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
        probs.append(torch.softmax(logits, 1).cpu())
    return total_loss / len(y), correct / len(y), torch.cat(probs)


def fit(model, cfg: TrainConfig, x_train, y_train, x_val, y_val, device, verbose: bool = True):
    """Train ``model`` and keep the weights of the epoch with the lowest validation loss."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(cfg, params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs) if cfg.cosine_schedule else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    hist = History()
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, run_correct = 0.0, 0
        for idx in iterate_batches(len(y_train), cfg.batch_size, shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            # A batch of 1024 does not fit into 8 GB on my laptop, so the batch is split into
            # micro-batches and the gradients are added up (gradient accumulation). The optimizer
            # still makes ONE step per batch of ``cfg.batch_size`` images.
            for part in idx.split(cfg.micro_batch):
                xb = prepare_batch(x_train[part], device, cfg.image_size, cfg.mean, cfg.std, cfg.augment)
                yb = y_train[part].to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                (loss * len(part) / len(idx)).backward()
                run_loss += loss.item() * len(part)
                run_correct += (logits.argmax(1) == yb).sum().item()
            opt.step()
        if sched:
            sched.step()
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, cfg, device)
        hist.train_loss.append(run_loss / len(y_train))
        hist.train_acc.append(run_correct / len(y_train))
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
        hist.epoch_time.append(time.perf_counter() - t0)
        if verbose:
            print(
                f"[{cfg.name}] epoch {epoch:2d}  train loss {hist.train_loss[-1]:.4f} acc {hist.train_acc[-1]:.4f}"
                f" | val loss {val_loss:.4f} acc {val_acc:.4f} | {hist.epoch_time[-1]:.1f}s"
            )
        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs, hist.best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"[{cfg.name}] early stopping, best epoch was {hist.best_epoch}")
                break
    hist.total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
