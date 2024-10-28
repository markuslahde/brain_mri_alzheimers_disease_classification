import torch as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(MyModel, self).__init__()  

        if backbone == 'resnet18':
            self.model = models.resnet18()
        elif backbone == 'resnet34':
            self.model = models.resnet34()
        else:
            self.model = models.resnet50()

    def forward(self, x):
        return self.model(x) 