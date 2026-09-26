"""Architecture from the assignment table; no BatchNorm is specified there."""
import torch
from torch import nn


class SmallCNN(nn.Sequential):
    def __init__(self):
        layers = []
        for i, (ci, co, k, stride) in enumerate([
            (3, 32, 7, 2), (32, 64, 5, 1), (64, 128, 3, 2),
            (128, 256, 1, 1), (256, 256, 3, 2), (256, 512, 1, 1),
        ]):
            layers += [nn.Conv2d(ci, co, k, stride, k // 2, bias=False),
                       nn.ReLU(inplace=True)]
            if i == 0:
                layers.append(nn.MaxPool2d(3, 2, 1))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                   nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 100)]
        super().__init__(*layers)


def configure():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def measured_flops(model, x):
    """PyTorch operator profiler: conv/linear MAC FLOPs, outside timing runs."""
    with torch.inference_mode(), torch.profiler.profile(with_flops=True) as prof:
        y = model(x)
    del y
    return sum(event.flops for event in prof.key_averages())
