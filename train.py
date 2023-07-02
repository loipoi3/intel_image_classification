import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from model import ResNet
from dataset import IntelDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim

SAVED_MODEL_PATH = './models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 100
LEARNING_RATE = 0.1
BATCH_SIZE = 128
ROOT_DIR_TRAIN = './data/seg_train'
ROOT_DIR_TEST = './data/seg_test'
LOAD_MODEL = False
PATH_TO_MODEL = None

def train_loop(model, criterion, optimizer, train_loader, scaler, losses):
    model.train()

    for idx, (data, targets) in enumerate(train_loader):
        # move X and y to device(gpu or cpu)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.cuda.amp.autocast():
            # making prediction
            pred = model(data)

            # calculate loss and append it to losses
            Loss = criterion(pred, targets)
            losses.append(Loss.item)


        # backward
        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()

def main():
    # if directory model exists than create this
    if not os.path.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

    # define data transformations
    transform = A.Compose([A.Resize(64, 64), A.ToRGB(), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Blur(p=0.3),
                           ToTensorV2()])

    # define model and move it to device(gpu or cpu)
    model = ResNet()
    model.to(DEVICE)

    # create dataset using train directory
    dataset = IntelDataset(root_dir=ROOT_DIR_TRAIN)

    # split images to train and val using dataset
    train_images, val_images = train_test_split(dataset.images, test_size=0.2, random_state=42)

    # create training dataloader
    train_dataset = IntelDataset(ROOT_DIR_TRAIN, transform=transform)
    train_dataset.images = train_images
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create validating dataloader
    val_dataset = IntelDataset(ROOT_DIR_TRAIN)
    val_dataset.images = val_images
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # create testing dataloader
    test_dataset = IntelDataset(ROOT_DIR_TEST)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # checking whether the model needs to be retrained
    if LOAD_MODEL:
        model = ResNet()
        model.to(DEVICE)
        model.load_state_dict(torch.load(PATH_TO_MODEL))

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    # define Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch}')

        # define losses for scheduler
        losses = []

        train_loop(model, criterion, optimizer, train_loader, scaler, losses)
        val_loop()

        # save model
        torch.save(model.state_dict(), SAVED_MODEL_PATH + f'/model{epoch}.pth')

        # calculate mean loss
        mean_loss = sum(losses) / len(losses)

        # update scheduler
        scheduler.step(mean_loss)

    test_fn()

if __name__ == '__main__':
    main()