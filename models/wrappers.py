
import torch
import torch.nn as nn

from models.hiera.hiera import HieraRaw

class BackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.loaded_params = []

    def forward(self, x):
        # hiera
        if isinstance(self.backbone, HieraRaw):
            x = self.backbone(x, return_intermediates=True)
        # mvitv2
        elif  type(self.backbone).__name__ == "MultiScaleVit":
            x = self.backbone.forward_features(x)
            x = [x]
        # resnextv2
        elif type(self.backbone).__name__ == "FeatureListNet":
            x = self.backbone(x)
        return x

    def load_my_state_dict(self, state_dict):
        own_state = self.backbone.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.Tensor):
                param = param.data
                self.loaded_params.append(name)
            own_state[name].copy_(param)
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True