from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import trange

from data import ImgDataLoaders
import model
import utils

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(self.model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100,
                   smooth_f=0.05, diverge_th=5):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)
        
        range_iterator = trange(num_iter, desc="Learning Rate Finder", leave=False)
        for iteration in range_iterator:

            loss = self._train_batch(iterator)

            lrs.append(lr_scheduler.get_last_lr()[0])

            # update lr
            lr_scheduler.step()
            
            # we save the exponentially weighted average of the losses, as they are noisy
            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                range_iterator.close()
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        self.model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r
                for base_lr in self.base_lrs]

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)

def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5, save_dir=".", model="model"):

    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    
    plt.savefig(str(Path(save_dir, f"lr_finder_{model}.jpg")))
    
def main(args):
    # system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 1234
    utils.setup_system(seed=SEED)
    
    model_name = "resnet"
    
    # data
    batch_size = 512
    train_dir = "data/CUB_200_2011/train/"
    test_dir = "data/CUB_200_2011/test/"
    dataloaders = ImgDataLoaders(train_dir, test_dir, batch_size, model_name=model_name)
    train_iterator, val_iterator = dataloaders.train_val_dataloader(valid_ratio=0.1)
    
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
    
    # lr cfg
    start_lr = 1e-7
    end_lr = 10
    num_iter = 100
    
    optimizer = optim.AdamW(trained_model.parameters(), lr=start_lr)

    # run
    lr_finder = LRFinder(trained_model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_iterator, end_lr, num_iter, diverge_th=5)
    
    plot_lr_finder(lrs, losses, model=model_name)
    
if __name__ == "__main__":
    main(args=None)