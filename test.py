from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from train import val_step
import utils
from data import ImgDataLoaders
import model

def test(args):
    # system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "resnet"
    saved_weights_path = Path(f"{model_name}_ckpt", f"best_{model_name}.pt")
    
    # data
    batch_size = 512
    train_dir = "data/CUB_200_2011/train/"
    test_dir = "data/CUB_200_2011/test/"
    dataloaders = ImgDataLoaders(train_dir, test_dir, batch_size, model_name=model_name)
    test_iterator = dataloaders.test_dataloader()
    
    # model and loss
    if model_name == "alexnet":
        test_model = model.alexnet_model(nb_classes=dataloaders.nb_classes, display_param_count=True)
    elif model_name == "vgg":
        test_model = model.vgg_model(nb_classes=dataloaders.nb_classes, display_param_count=True)
    elif model_name == "resnet":
        test_model = model.resnet_model(nb_classes=dataloaders.nb_classes, display_param_count=True)
    elif model_name == "convnext":
        test_model = model.convnext_model(nb_classes=dataloaders.nb_classes, display_param_count=True)
    elif model_name == "efficientnet":
        test_model = model.efficientnet_model(nb_classes=dataloaders.nb_classes, display_param_count=True)
        
    test_model.load_state_dict(torch.load(str(saved_weights_path)))
    criterion = nn.CrossEntropyLoss()
    
    test_model = test_model.to(device)
    criterion = criterion.to(device)
    
    # metrics
    test_loss_metric = utils.Metric("Val. Loss", float('inf'))
    test_acc_1_metric = utils.Metric("Val. Acc @1", 0, percentage=True)
    
    # test
    test_start_time = time.monotonic()
    test_loss, test_acc_1 = val_step(test_model, test_iterator, criterion, device)
    test_end_time = time.monotonic()
    
    test_loss_metric.set_value(test_loss)
    test_acc_1_metric.set_value(test_acc_1*100)
    utils.print_status_bar(step=1, total=1, 
                    start_time=test_start_time, end_time=test_end_time, 
                    loss=test_loss_metric, metrics=[test_acc_1_metric], delim=" | ") 
    
if __name__ == "__main__":
    test(args=None)