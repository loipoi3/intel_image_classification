from torchvision.models import resnet152
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # download pretrained resnet152
        self.resnet = resnet152(pretrained=True)

        # freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # change out_features in fc from 1000 to 6
        self.resnet.fc = nn.Linear(in_features=2048, out_features=6, bias=True)
        print(self.resnet)

    def forward(self, x):
        # make prediction
        pred = self.resnet(x)

        return pred