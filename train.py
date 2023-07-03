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
from utils import calculate_accuracy

SAVED_MODEL_PATH = './models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
LEARNING_RATE = 0.1
BATCH_SIZE = 1024
ROOT_DIR_TRAIN = './data/seg_train'
ROOT_DIR_TEST = './data/seg_test'
LOAD_MODEL = False
PATH_TO_MODEL = None

def train_loop(model, criterion, optimizer, train_loader, scaler, losses, accuracies):
    model.train()

    for idx, (data, targets) in enumerate(train_loader):
        # move data and targets to device(gpu or cpu)
        data = data['image'].to(DEVICE).float()
        targets = targets.to(DEVICE)

        with torch.cuda.amp.autocast():
            # making prediction
            pred = model(data)

            # calculate loss and accuracy and append it to losses and accuracies
            Loss = criterion(pred, targets)
            losses.append(Loss.item())
            accuracy = calculate_accuracy(pred, targets)
            accuracies.append(accuracy)

        # backward
        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()

def val_loop(model, criterion, val_loader, losses, accuracies):
    model.eval()

    with torch.no_grad():
        for idx, (data, targets) in enumerate(val_loader):
            # move data and targets to device(gpu or cpu)
            data = data['image'].to(DEVICE).float()
            targets = targets.to(DEVICE)

            # making prediction
            pred = model(data)

            # calculate loss and accuracy and append it to losses and accuracies
            Loss = criterion(pred, targets)
            losses.append(Loss.item())
            accuracy = calculate_accuracy(pred, targets)
            accuracies.append(accuracy)

def test_fn(model, criterion, test_loader, losses, accuracies):
    model.eval()

    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            # move data and targets to device(gpu or cpu)
            data = data['image'].to(DEVICE).float()
            targets = targets.to(DEVICE)

            # making prediction
            pred = model(data)

            # calculate loss and accuracy and append it to losses and accuracies
            Loss = criterion(pred, targets)
            losses.append(Loss.item())
            accuracy = calculate_accuracy(pred, targets)
            accuracies.append(accuracy)

def main():
    # if directory model exists than create this
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

    # define data transformations
    transform_train = A.Compose([A.Resize(64, 64), A.ToRGB(), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Blur(p=0.3),
                           ToTensorV2()])
    transform_val_test = A.Compose([A.Resize(64, 64), ToTensorV2()])

    # define model and move it to device(gpu or cpu)
    model = ResNet()
    model.to(DEVICE)

    # create dataset using train directory
    dataset = IntelDataset(root_dir=ROOT_DIR_TRAIN)

    # split images to train and val using dataset
    train_images, val_images = train_test_split(dataset.images, test_size=0.2, random_state=42)

    # create training dataloader
    train_dataset = IntelDataset(ROOT_DIR_TRAIN, transform=transform_train)
    train_dataset.images = train_images
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create validating dataloader
    val_dataset = IntelDataset(ROOT_DIR_TRAIN, transform=transform_val_test)
    val_dataset.images = val_images
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # create testing dataloader
    test_dataset = IntelDataset(ROOT_DIR_TEST, transform=transform_val_test)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch}')

        # define losses for scheduler and accuracies
        losses_train = []
        accuracies_train = []

        train_loop(model, criterion, optimizer, train_loader, scaler, losses_train, accuracies_train)

        # calculate mean loss and accuracy on train
        mean_loss_train = sum(losses_train) / len(losses_train)
        mean_accuracy_train = sum(accuracies_train) / len(accuracies_train)

        print(f"Training Loss: {mean_loss_train} | Training Accuracy: {mean_accuracy_train}")

        losses_val = []
        accuracies_val = []

        val_loop(model, criterion, val_loader, losses_val, accuracies_val)

        # calculate mean loss and accuracy on val
        mean_loss_val = sum(losses_val) / len(losses_val)
        mean_accuracy_val = sum(accuracies_val) / len(accuracies_val)

        print(f"Validating Loss: {mean_loss_val} | Validating Accuracy: {mean_accuracy_val}")

        losses_test = []
        accuracies_test = []

        test_fn(model, criterion, test_loader, losses_test, accuracies_test)

        # calculate mean loss and accuracy on test
        mean_loss_test = sum(losses_test) / len(losses_test)
        mean_accuracy_test = sum(accuracies_test) / len(accuracies_test)

        print(f"Testing Loss: {mean_loss_test} | Testing Accuracy: {mean_accuracy_test}")

        # save model
        torch.save(model.state_dict(), SAVED_MODEL_PATH + f'/model{epoch}.pth')

        # update scheduler
        scheduler.step(mean_loss_train)

if __name__ == '__main__':
    main()