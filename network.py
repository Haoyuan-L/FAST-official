import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=True),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNN4Conv(nn.Module):
    def __init__(self, in_channels, num_classes, img_size):
        super(CNN4Conv, self).__init__()
        in_channels = in_channels
        num_classes = num_classes
        hidden_size = 64
        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        if img_size == 64:
            self.emb_dim = hidden_size * 4 * 4
        elif img_size == 32:
            self.emb_dim = hidden_size * 2 * 2
        elif img_size == 28:
            self.emb_dim = hidden_size * 1 * 1
        else:
            raise NotImplementedError(f"Unsupported image size: {img_size}")

        # Compute emb_dim
#        with torch.no_grad():
#            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
#            output_feat = self.features(dummy_input)
#            self.emb_dim = output_feat.view(1, -1).size(1)

        self.linear = nn.Linear(self.emb_dim, num_classes)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)
        
        return logits
    
    def get_embedding_dim(self):
        return self.emb_dim

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
       # embedding_output = out.view(out.size(0), -1)
       # print(embedding_output.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        embedding_output = out
        out = self.linear(out)
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

class LinearModelMulti(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=1, dropout_rate=0.3):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6), 
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(hidden_size, hidden_size),
                                            torch.nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6),
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(hidden_size, num_classes))
    def forward(self, x):
        return self.linear(x)

class LinearProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
""" Helper function to get the network model """

def get_resnet18_network(input_shape=None, num_classes=10, weights_fp=None):
    model = ResNet18(num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model

def get_cnn4_network(input_shape, num_classes, weights_fp=None):
    in_channels = input_shape[0]
    img_size = input_shape[1]
    model = CNN4Conv(in_channels, num_classes, img_size)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model

def get_linear_network(input_shape, num_classes, weights_fp=None):
    """
    Initializes the LinearClassifier model.
    """
    emb_dim = input_shape[0]
    model = LinearProbe(embedding_dim=emb_dim, num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model
