import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from vit_pytorch import ViT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 100,
    dim = 512,
    depth = 4,
    heads = 12,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)
# v = models.resnet18()
img = torch.randn(1, 3, 224, 224)
print(v) 
# preds = v(img) # (1, 1000)

num_params = count_parameters(v)
print(f'The vit model has {num_params:,} trainable parameters')