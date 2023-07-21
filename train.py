import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import utils
from data import ImgDataLoaders
import model

def train_step(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    
    model.train()
    
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1 = utils.calculate_topk_accuracy(y_pred, y, k=1)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
        
    return epoch_loss, epoch_acc_1

def val_step(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
        
            acc_1 = utils.calculate_topk_accuracy(y_pred, y, k=1)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
        
    return epoch_loss, epoch_acc_1

def train(args):
    # system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 1234
    utils.setup_system(seed=SEED)
    
    model_name = "resnet"
    save_dir = f"{model_name}_ckpt"
    utils.create_dir(save_dir)
    
    # data
    batch_size = 512
    train_dir = "data/CUB_200_2011/train/"
    test_dir = "data/CUB_200_2011/test/"
    dataloaders = ImgDataLoaders(train_dir, test_dir, batch_size, model_name=model_name)
    train_iterator, val_iterator = dataloaders.train_val_dataloader(valid_ratio=0.1)
    
    with open(Path(save_dir, 'classes.json'), 'w') as writer:
        json.dump(dataloaders.image_classes, writer)
    
    # model and loss
    if model_name == "alexnet":
        trained_model = model.alexnet_model(nb_classes=dataloaders.nb_classes)
    elif model_name == "vgg":
        trained_model = model.vgg_model(nb_classes=dataloaders.nb_classes)
    elif model_name == "resnet":
        trained_model = model.resnet_model(nb_classes=dataloaders.nb_classes)
    elif model_name == "convnext":
        trained_model = model.convnext_model(nb_classes=dataloaders.nb_classes)
    elif model_name == "efficientnet":
        trained_model = model.efficientnet_model(nb_classes=dataloaders.nb_classes)
    
    criterion = nn.CrossEntropyLoss()
    
    trained_model = trained_model.to(device)
    criterion = criterion.to(device)
    
    # train cfg
    lr = 1.5e-3
    optimizer = optim.AdamW(trained_model.parameters(), lr=lr)
    epochs = 30
    
    # metrics 
    train_loss_metric = utils.Metric("Train Loss", float('inf'))
    train_acc_1_metric = utils.Metric("Train Acc @1", 0, percentage=True)
    val_loss_metric = utils.Metric("Val. Loss", float('inf'))
    val_acc_1_metric = utils.Metric("Val. Acc @1", 0, percentage=True)

    # train
    best_weight = utils.BestWeightSaving(save_dir=save_dir, loss=True)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.monotonic()
        
        train_loss, train_acc_1 = train_step(trained_model, train_iterator, optimizer, criterion, device)
        val_loss, val_acc_1 = val_step(trained_model, val_iterator, criterion, device)
        
        best_weight.step(trained_model, val_loss, name=model_name)
        
        epoch_end_time = time.monotonic()
        
        train_loss_metric.set_value(train_loss)
        train_acc_1_metric.set_value(train_acc_1*100)
        val_loss_metric.set_value(val_loss)
        val_acc_1_metric.set_value(val_acc_1*100)
        utils.print_status_bar(step=epoch, total=epochs, 
                        start_time=epoch_start_time, end_time=epoch_end_time, 
                        loss=train_loss_metric, 
                        metrics=[train_acc_1_metric, val_loss_metric, val_acc_1_metric], 
                        delim=" | ") 

        
if __name__ == "__main__":
    train(args=None)