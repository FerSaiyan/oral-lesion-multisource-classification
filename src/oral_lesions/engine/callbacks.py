# src/oral_lesions/engine/callbacks.py
import torch

class CheckpointSaver:
    def __init__(self): self.best = -1.0
    def update(self, model, score, path):
        if score > self.best:
            self.best = score; torch.save(model.state_dict(), path); print(f"  saved best → {path}")

class EarlyStopping:
    def __init__(self, patience=7): self.patience, self.bad = patience, 0
    def update(self, model, score, path):
        if not hasattr(self, "best"): self.best = -1.0
        if score > self.best: self.best, self.bad = score, 0
        else: self.bad += 1
        if self.bad >= self.patience: raise StopIteration("early stop")
