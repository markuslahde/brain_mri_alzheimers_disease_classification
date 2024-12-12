import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

class ModelResnet18(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=4):
        super(ModelResnet18, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResnet34(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=4):
        super(ModelResnet34, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet34(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResnet50(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=4):
        super(ModelResnet50, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)