from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def print_nb_param(model_name, nb):
    print(f'{model_name} model`s feature part has {nb:,} trainable parameters')
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def classifier_head(input_dim, output_dim):
    """
    """
    return nn.Sequential(
        # nn.Linear(in_features=input_dim, out_features=1024, bias=True),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.3),
        nn.Linear(in_features=input_dim, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1024, out_features=output_dim, bias=True)
    )

alexnet_weights = models.AlexNet_Weights.IMAGENET1K_V1
def alexnet_model(nb_classes=10, display_param_count=False):
    """Alexnet train model
    features
    avgpool
    classifier
    """
    
    model = models.alexnet(weights=alexnet_weights)
    classifier_input_dim = model.classifier[1].in_features
    classifier = classifier_head(input_dim=classifier_input_dim, output_dim=nb_classes)
    model.classifier = classifier
    
    # model's feature parameters count
    if display_param_count:
        total_param_count = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
        print_nb_param("alexnet", total_param_count)
    
    return model

vgg_weights = models.VGG11_BN_Weights.IMAGENET1K_V1
def vgg_model(nb_classes=10, display_param_count=False):
    """VGG 11 model
    """
    model = models.vgg11_bn(weights=vgg_weights)

    # model's feature parameters count
    if display_param_count:
        total_param_count = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
        print_nb_param("vgg 11 with bn", total_param_count)
        
    return model

resnet_weights = models.ResNet18_Weights.IMAGENET1K_V1
def resnet_model(nb_classes=10, feature_pretrained=True, display_param_count=False):
    """Resnet 18 model
    
    [stem]
    layer1 --> layer4
    avgpool
    fc
    """
    weights = None
    if feature_pretrained:
        weights = resnet_weights
    model = models.resnet18(weights=weights)
    
    classifier_input_dim = model.fc.in_features
    classifier = classifier_head(input_dim=classifier_input_dim, output_dim=nb_classes)
    model.fc = classifier
    
    # model's feature parameters count
    if display_param_count:
        total_param_count = 0
        total_param_count += sum(p.numel() for p in model.conv1.parameters() if p.requires_grad)
        total_param_count += sum(p.numel() for p in model.bn1.parameters() if p.requires_grad)
        total_param_count += sum(p.numel() for p in model.layer1.parameters() if p.requires_grad)
        total_param_count += sum(p.numel() for p in model.layer2.parameters() if p.requires_grad)
        total_param_count += sum(p.numel() for p in model.layer3.parameters() if p.requires_grad)
        total_param_count += sum(p.numel() for p in model.layer4.parameters() if p.requires_grad)
        
        print_nb_param("resnet18", total_param_count)
        
    return model

def classifier_head_convnet(input_dim, output_dim):
    """
    """
    norm_layer = partial(LayerNorm2d, eps=1e-6)
    
    return nn.Sequential(
        # nn.Linear(in_features=input_dim, out_features=1024, bias=True),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.3),
        norm_layer(input_dim),
        nn.Flatten(1),
        nn.Linear(in_features=input_dim, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1024, out_features=output_dim, bias=True)
    )
    
convnext_weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
def convnext_model(nb_classes=10, feature_pretrained=True, display_param_count=False):
    """ConvNext base model
    
    This model needs a specific classifier head, as its classifier part has a layernorm2d
    followed by a flatten.
    
    [stem]
    stage1 -> stage4
    avgpool
    classifier(begins with layer normalization 2d)
    """
    weights = None
    if feature_pretrained:
        weights = convnext_weights
    model = models.convnext_base(weights=weights)
    
    classifier_input_dim = model.classifier[2].in_features
    classifier = classifier_head_convnet(input_dim=classifier_input_dim, 
                                         output_dim=nb_classes)
    model.classifier = classifier
    
    # model's feature parameters count
    if display_param_count:
        total_param_count = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
        print_nb_param("convnext base", total_param_count)
    
    return model

efficientnet_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
def efficientnet_model(nb_classes=10, feature_pretrained=True, display_param_count=False):
    """Efficientnet B0 model
    
    This model needs a specific classifier head, as its classifier part has a layernorm2d
    followed by a flatten.
    
    conv 3x3
    stage1 -> stage7
    conv 1x1
    avgpool
    classifier
    """
    weights = None
    if feature_pretrained:
        weights = efficientnet_weights
    model = models.efficientnet_b0(weights=weights)
    
    classifier_input_dim = model.classifier[1].in_features
    classifier = classifier_head(input_dim=classifier_input_dim, output_dim=nb_classes)
    model.classifier = classifier
    
    # model's feature parameters count
    if display_param_count:
        total_param_count = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
        print_nb_param("efficientnet b0", total_param_count)
    
    return model

if __name__ == "__main__":
    m = resnet_model(10, display_param_count=True)
    # print(m)
    
    # input_dummy = torch.randn((1,3,300,300))
    # m.eval()
    # with torch.no_grad():
    #     print(m(input_dummy))