import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from model import ResNet
from dataset import IntelDataset
from sklearn.model_selection import train_test_split

SAVED_MODEL_PATH = './models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
ROOT_DIR_TRAIN = './data/seg_train'

def main():
    # if directory model exists than create this
    if not os.path.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

    # define data transformations
    transform = A.Compose([A.Resize(64, 64), A.ToRGB(), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Blur(p=0.3),
                           ToTensorV2()])
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define model and move it to device(gpu or cpu)
    model = ResNet()
    model.to(DEVICE)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)





if __name__ == '__main__':
    main()