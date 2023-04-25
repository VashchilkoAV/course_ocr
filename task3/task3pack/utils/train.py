import sys
import warnings

import numpy as np

import torch
from torch import optim as optim

from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_train_plots(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def train(model, criterion, device, train_dataset, test_dataset, train_dataloader_kwargs, test_dataloader_kwargs, training_kwargs, wandb_instance=None, eval_every=5):
    test_losses = []
    train_losses = []

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_kwargs['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, training_kwargs['milestones'], gamma=training_kwargs['gamma'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_kwargs)
    
    # first eval
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for input, target in test_dataloader:
            input, target = input.to(device), target.to(device)
            pred = model(input)
            loss = criterion(pred, target)
            loss_sum += loss.item()

        val_loss = loss_sum / len(test_dataset)
        test_losses.append(val_loss)
        print(f'Initial val loss: {val_loss}')
        if wandb_instance is not None:
            wandb_instance.log({
                'val': {
                    'loss': val_loss,
                },
            }, step=0)

    # loop
    for epoch in range(training_kwargs['epochs']):
        print(f'Epoch {epoch + 1}:')

        # train
        model.train()
        loss_sum = 0.
        for input, target in train_dataloader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(input)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss = loss_sum / len(train_dataset)
        train_losses.append(train_loss)
        print(f'Train loss: {train_loss}')
        if wandb_instance is not None:
            wandb_instance.log({
                'train': {
                    'loss': train_loss,
                },
                'lr': scheduler.get_last_lr()[0],
            }, step=epoch+1)


        scheduler.step()

        if (epoch + 1) % eval_every == 0:
        # eval
            model.eval()
            with torch.no_grad():
                loss_sum = 0.
                for input, target in test_dataloader:
                    input, target = input.to(device), target.to(device)
                    pred = model(input)
                    loss = criterion(pred, target)
                    loss_sum += loss.item()

                val_loss = loss_sum / len(test_dataset)
                test_losses.append(val_loss)
                print(f'Val loss: {val_loss}')
                if wandb_instance is not None:
                    wandb_instance.log({
                        'val': {
                            'loss': val_loss,
                        },
                    }, step=epoch+1)

    return np.array(train_losses), np.array(test_losses), model