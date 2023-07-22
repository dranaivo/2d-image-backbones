from pathlib import Path
import random

import numpy as np
import torch

### Utils
def create_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    
### System
def setup_system(seed=1234):
    '''System setup for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cudnn_benchmark_enabled: enable CuDNN benchmark for the sake of performance
        torch.backends.cudnn_benchmark_enabled = True
        # cudnn_deterministic: make cudnn deterministic (reproducible training)
        torch.backends.cudnn.deterministic = True

### Metrics
def calculate_topk_accuracy(y_pred, y, k=5):
    """
    returns
        When k=1, returns the accuracy.
    """
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_k = correct_k / batch_size
    return acc_k

def calculate_topk_per_class_accuracy(y_pred, y, k=5, idx=0):
    # y_pred_sliced = y_pred[specific class]
    # class_accuracy = calculate_topk_accuracy(y_pred_sliced, y_sliced, k=5)
    return 0

### Callbacks
class BestWeightSaving:
    def __init__(self, save_dir, loss=True):
        
        self.save_dir = save_dir
        create_dir(self.save_dir)
        
        self.loss = loss
        if self.loss:
            self.best = float("inf")
        else:
            self.best = 0
        
    def step(self, model, value, name="model"):
        if self.loss:
            if value < self.best:
                self.best = value
                torch.save(model.state_dict(), str(Path(self.save_dir, f"best_{name}.pt")))
        else:
            if value > self.best:
                self.best = value
                torch.save(model.state_dict(), str(Path(self.save_dir, f"best_{name}.pt")))

### Progress report
def time_mins_secs(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Metric:
    def __init__(self, name, value, percentage=False):
        self.name = name
        self.value = value
        self.percentage = percentage
    def is_percentage(self):
        return self.percentage
    def set_value(self, value):
        self.value = value
    def result(self):
        return self.value
    
def print_status_bar(step, total, start_time, end_time, 
                        loss, metrics=None, delim=" - "):
    
    mins, secs = time_mins_secs(start_time, end_time)
    
    # each metric has a .name attribute and .result() method
    metrics = delim.join([f"{m.name}: {m.result():.01f} %" if m.is_percentage() else 
                          f"{m.name}: {m.result():.03f}"
                          for m in [loss] + (metrics or [])])
    
    end = "\n" 
    print(f"Epoch {step:02}/{total:02} - Time: {mins:02}m {secs:02}s{delim}" + metrics, end=end)