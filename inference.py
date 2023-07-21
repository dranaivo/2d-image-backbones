from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as T

from train import val_step
import utils
import model

class InferenceModel(nn.Module):
    def __init__(self, model_weights, model_forward):
        super().__init__()
        self.model_preprocess = model_weights.transforms()
        self.model_forward = model_forward
        self.model_forward.eval()
        
    def pre_process(self, x):
        """
        args
            x : Tensor, placed on device 
        """
        x = self.model_preprocess(x)
        x.unsqueeze_(0)
        return x

    def post_process(self, logits):
        prob = F.softmax(logits, dim=-1)
        result = prob.max(1, keepdim=True)
        top_pred = result.indices
        top_prob = result.values
        
        return top_pred, top_prob, prob

    def forward(self, img):
        with torch.no_grad():
            img_pre = self.pre_process(img)
            logits = self.model_forward(img_pre)   
            y_top, y_prob, all_prob = self.post_process(logits)
        
        return y_top, y_prob, all_prob

def performance_ms_fps(nb, start_time, end_time):
    elapsed_time = (end_time - start_time) / nb
    time_ms = elapsed_time * 1000
    fps = 1 / elapsed_time
    return time_ms, fps

def predict(args):
    
    # system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "resnet"
    save_dir = f"{model_name}_ckpt"
    
    with open(Path(save_dir, 'classes.json'), 'r') as json_file:
        classes = json.load(json_file)
    nb_classes = len(classes)
    
    if model_name == "alexnet":
        alexnet_model_forward = model.alexnet_model(nb_classes=nb_classes, 
                                              display_param_count=True)
        alexnet_model_forward.load_state_dict(torch.load(
            str(Path(save_dir, f"best_{model_name}.pt"))
        ))
        # inference model
        inference_model = InferenceModel(model_weights=model.alexnet_weights, 
                                         model_forward=alexnet_model_forward)
    elif model_name == "vgg":
        vgg_model_forward = model.vgg_model(nb_classes=nb_classes, display_param_count=True)
        vgg_model_forward.load_state_dict(torch.load(
            str(Path(save_dir, f"best_{model_name}.pt"))
        ))
        # inference model
        inference_model = InferenceModel(model_weights=model.vgg_weights, 
                                         model_forward=vgg_model_forward)
    elif model_name == "resnet":
        resnet_model_forward = model.resnet_model(nb_classes=nb_classes,
                                           display_param_count=True) 
        resnet_model_forward.load_state_dict(torch.load(
            str(Path(save_dir, f"best_{model_name}.pt"))
        ))
        # inference model
        inference_model = InferenceModel(model_weights=model.resnet_weights, 
                                         model_forward=resnet_model_forward)
    elif model_name == "convnext":
        convnext_model_forward = model.convnext_model(nb_classes=nb_classes,
                                                    display_param_count=True) 
        convnext_model_forward.load_state_dict(torch.load(
            str(Path(save_dir, f"best_{model_name}.pt"))
        ))
        # inference model
        inference_model = InferenceModel(model_weights=model.convnext_weights, 
                                         model_forward=convnext_model_forward)
    elif model_name == "efficientnet":
        efficientnet_model_forward = model.efficientnet_model(nb_classes=nb_classes,
                                                          display_param_count=True) 
        efficientnet_model_forward.load_state_dict(torch.load(
            str(Path(save_dir, f"best_{model_name}.pt"))
        ))
        # inference model
        inference_model = InferenceModel(model_weights=model.efficientnet_weights, 
                                         model_forward=efficientnet_model_forward)
    
    if True: # timing
        # TODO : max size of (ds)
        input_dummy = torch.randint(low=0, high=255, 
                                    size=(3,1500,1500), dtype=torch.uint8) 
        iteration = 2500

        # warmup
        _ = inference_model(input_dummy) 

        inferences_start_time = time.monotonic()
        for _ in range(iteration): inference_model(input_dummy)
        inferences_end_time = time.monotonic()

        perf_ms, fps = performance_ms_fps(iteration, 
                                          inferences_start_time, inferences_end_time)
        print(f"end-to-end {perf_ms:.01f} ms, {fps:.01f} frame/s")
        
    if False: # predict
        y = "129.Song_Sparrow"
        for img in Path(f"data/CUB_200_2011/test/{y}/").glob("*.jpg"):
            input_img = Image.open(str(img))
            input_img = T.pil_to_tensor(input_img)

            # inference
            input_img = input_img.to(device)
            inference_model.to(device)

            y_top, y_prob, _ = inference_model(input_img)

            # plot
            print("gt: {}, pred: {}, prob: {:.02f}".format(
                y,
                classes[y_top.cpu().detach().item()],
                y_prob.cpu().detach().item()
            ))
    
if __name__ == "__main__":
    from PIL import Image
    predict(args=None)