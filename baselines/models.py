import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import CNN4Conv

def get_model(args):
    """Get model based on configuration."""
    if args.model == 'cnn4conv':
        return CNN4Conv(args.in_channels, args.num_classes, args.img_size)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
        if args.in_channels != 3:
            model.conv1 = nn.Conv2d(args.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        return model
    elif args.model == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        if args.in_channels != 3:
            model.features[0][0] = nn.Conv2d(args.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {args.model}")
