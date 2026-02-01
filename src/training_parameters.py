from dataclasses import dataclass
import torch
import segmentation_models_pytorch as smp

@dataclass
class TrainingParameters:
    epochs: str
    devcie: torch.device
    metrics: list
    optimizer: torch.optim
    loss: smp.losses
    scheduler: torch.optim
    warmup_epochs: int
