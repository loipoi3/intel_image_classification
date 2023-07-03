from torchvision.models import resnet18
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # download pretrained resnet50
        self.resnet = resnet18(weights='ResNet18_Weights.DEFAULT')

        # freeze all layers
        #for param in self.resnet.parameters():
        #    param.requires_grad = False

        # find index for layer4
        #layer4_index = None
        #for i, module in enumerate(self.resnet.modules()):
        #    if module == self.resnet.layer4:
        #        layer4_index = i
        #        break

        # unfreeze all layers started from layer4
        #for module in list(self.resnet.modules())[layer4_index:]:
        #    for param in module.parameters():
        #        param.requires_grad = True

        # change out_features in fc from 1000 to 6
        self.resnet.fc = nn.Linear(in_features=512, out_features=6, bias=True)

    def forward(self, x):
        # make prediction
        pred = self.resnet(x)

        return pred

model = ResNet()