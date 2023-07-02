import torch

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy