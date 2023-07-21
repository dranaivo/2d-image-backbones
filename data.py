"""
All transforms in weights follow https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38 :

class ImageClassification(nn.Module):
    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, 
            antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img
"""
import copy

from PIL import Image
import torch
import torch.utils.data as data
from torchvision import models, transforms, datasets

import model


class ImgDataLoaders():
    def __init__(self, train_dir, test_dir, batch_size=128, model_name="resnet"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_classes = datasets.ImageFolder(root=self.test_dir).classes
        self.nb_classes = len(self.image_classes)
        self.batch_size = batch_size
        
        if model_name == "alexnet":
            self.weights = model.alexnet_weights
        elif model_name == "vgg":
            self.weights = model.vgg_weights
        elif model_name == "resnet":
            self.weights = model.resnet_weights
        elif model_name == "convnext":
            self.weights = model.convnext_weights
        elif model_name == "efficientnet":
            self.weights = model.efficientnet_weights
        
        # some data aug 
        self.train_transforms = transforms.Compose([
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            self.weights.transforms()
                        ])
        self.test_transforms = self.weights.transforms()
            
    def train_val_dataloader(self, valid_ratio=0.1):
        ds = datasets.ImageFolder(root=self.train_dir, transform=self.train_transforms)

        # train/val split
        n_valid_examples = int(len(ds) * valid_ratio)
        n_train_examples = len(ds) - n_valid_examples

        train_data, valid_data = data.random_split(ds, 
                                                   [n_train_examples, n_valid_examples])

        # overwrite val transforms with test transforms, as we don't want data augmentation
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = self.test_transforms

        # dataloaders
        train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_dataloader = data.DataLoader(valid_data, batch_size=self.batch_size)

        return train_dataloader, valid_dataloader

    def test_dataloader(self):
        """
        """
        ds = datasets.ImageFolder(root=self.test_dir, transform=self.test_transforms)

        # dataloader
        test_dataloader = data.DataLoader(ds, batch_size=self.batch_size)

        return test_dataloader
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    test_data_dir = "data/CUB_200_2011/train/"
    train_data_dir = "data/CUB_200_2011/test/"
    dataloaders = ImgDataLoaders(train_data_dir, test_data_dir, 8, "resnet")
    iterator = dataloaders.test_dataloader()
    print(dataloaders.nb_classes)
    
    def normalize_image(image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min = image_min, max = image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image  

    plt.imshow(normalize_image(next(iter(iterator))[0][0]).permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.savefig("image.jpg")