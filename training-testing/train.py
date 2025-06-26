import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dns.dnssec import validate
from ray.air.examples.custom_trainer import train_dataset
from torch.utils.data import DataLoader
from dataloader import DataLoaderNoisyClean, dataset,transformStruct
import hydra
from omegaconf import DictConfig
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray import train
from ray.tune.search import ConcurrencyLimiter
import numpy as np
import mlflow
import logging
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

#import models
from models import Unet







#pipeline
#https://hydra.cc/docs/plugins/optuna_sweeper/
#https://hydra.cc/docs/plugins/ray_launcher/

@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    #import data set
    processedDataSet = DataLoaderNoisyClean(dataset, transform=transformStruct)
    train_dataset, validate_dataset,test_dataset = torch.utils.data.random_split(processedDataSet, [cfg.train.train,cfg.train.validation,cfg.train.test]) #must sum to 1 (using percentages)

    #Starting mlflow to log
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            'modelName': cfg.model.name,
            'learningRate' : cfg.optimizer.lr,
            'weightDecay' : cfg.optimizer.weight_decay,
            'batchSize' : cfg.model.batch_size,
            'epochs' : cfg.train.epochs,
        })

        model = hydra.utils.instantiate(cfg.model)
        optimizer = hydra.utils.instantiate(cfg.optimizer,params=model.parameters())
        criterion = hydra.utils.instantiate(cfg.train.loss)
        #Implement training logic
        for epochs in range(cfg.train.epochs):
            return 0






