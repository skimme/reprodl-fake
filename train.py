#!/usr/bin/env python
# coding: utf-8

from pytorch_lightning import loggers
import torch
import torchaudio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.metrics import functional
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import wandb


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path = Path("data\ESC-50-master\ESC-50-master"), 
                 sample_rate: int = 8000,
                 folds = [1]):
        # Load CSV & initialize all torchaudio.transforms
        # Resample --> MelSpectrogram --> AmplitudeToDB     
        self.path = path
        self.csv = pd.read_csv(path / Path("meta/esc50.csv"))
        self.csv = self.csv[self.csv["fold"].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __getitem__(self, index):
        # Returns (xb, yb) pair
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / "audio" / row["filename"])
        label = row["target"]
        xb = self.db(self.melspec(self.resample(wav)))
        return xb, label

    def __len__(self):
        # Return Length
        return len(self.csv)


class AudioNet(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.hp = hparams.model
        self.conv1 = nn.Conv2d(1, self.hp.base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(self.hp.base_filters)
        self.conv2 = nn.Conv2d(self.hp.base_filters, self.hp.base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.hp.base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(self.hp.base_filters, self.hp.base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.hp.base_filters * 2)
        self.conv4 = nn.Conv2d(self.hp.base_filters * 2, self.hp.base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.hp.base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.hp.base_filters * 4, self.hp.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.optim.lr)
        return optimizer

@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):    
    path = Path(get_original_cwd()) / Path(cfg.data.path)
    train_data = ESC50Dataset(path=path, folds=cfg.data.train_folds)
    val_data = ESC50Dataset(path=path, folds=cfg.data.val_folds)
    test_data = ESC50Dataset(path=path, folds=cfg.data.test_folds)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size, num_workers=0)

    pl.seed_everything(cfg.seed)

    wanb_logger = pl.loggers.WandbLogger(project="reorodl")

    audionet = AudioNet(cfg)
    trainer = pl.Trainer(**cfg.trainer, logger=wanb_logger)
    trainer.fit(audionet, train_loader, val_loader)

    
if __name__ == "__main__":
    train()