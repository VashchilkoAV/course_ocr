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


def train_new(model, criterion, device, train_dataset, test_dataset, train_dataloader_kwargs, test_dataloader_kwargs, training_kwargs, wandb_instance=None, eval_every=5):
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


def train_with_trainable_loss(model, criterion, trainable_criterion, device, train_dataset, test_dataset, train_dataloader_kwargs, test_dataloader_kwargs, training_kwargs, wandb_instance=None, eval_every=5):
    test_regular_losses = []
    train_regular_losses = []
    test_trainable_losses = []
    train_trainable_losses = []

    model.to(device)
    trainable_criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_kwargs['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, training_kwargs['milestones'], gamma=training_kwargs['gamma'])
    criterion_optimizer = torch.optim.SGD(trainable_criterion.parameters(), lr=training_kwargs['lr_criterion'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_kwargs)
    
    # first eval
    model.eval()
    with torch.no_grad():
        regular_loss_sum = 0.
        trainable_loss_sum = 0.
        n_good = 0
        n_all = 0
        for input, target in test_dataloader:
            input, target = input.to(device), target.to(device)
            pred, feat = model(input)
            
            regular_loss = criterion(pred, target)
            trainable_loss = trainable_criterion(feat, target) * training_kwargs['weight_criterion']
            
            classes = torch.argmax(pred, dim=1).cpu().numpy()
            n_good += sum(classes == target.cpu().numpy())
            n_all += len(classes)
            
            regular_loss_sum += regular_loss.item()
            trainable_loss_sum += trainable_loss.item()

        val_regular_loss = regular_loss_sum / len(test_dataset)
        val_trainable_loss = trainable_loss_sum / len(test_dataset)
        test_regular_losses.append(val_regular_loss)
        test_trainable_losses.append(val_trainable_loss)

        print(f'Initial val: [regular_loss: {val_regular_loss}, trainable_loss: {val_trainable_loss}, accuracy: {n_good / n_all}]')
        if wandb_instance is not None:
            wandb_instance.log({
                'val': {
                    'regular_loss': val_regular_loss,
                    'trainable_loss': val_trainable_loss,
                    'accuracy': n_good / n_all,
                },
            }, step=0)

    # loop
    for epoch in range(training_kwargs['epochs']):
        print(f'Epoch {epoch + 1}:')

        # train
        model.train()
        regular_loss_sum = 0.
        trainable_loss_sum = 0.
        for input, target in train_dataloader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            criterion_optimizer.zero_grad()
            pred, feat = model(input)
            regular_loss = criterion(pred, target)
            trainable_loss = trainable_criterion(feat, target) * training_kwargs['weight_criterion']
            regular_loss_sum += regular_loss.item()
            trainable_loss_sum += trainable_loss.item()

            loss = regular_loss + trainable_loss
            loss.backward()

            optimizer.step()
            for param in trainable_criterion.parameters():
                param.grad.data *= (1. / training_kwargs['weight_criterion'])
            criterion_optimizer.step()

        train_regular_loss = regular_loss_sum / len(train_dataset)
        train_trainable_loss = trainable_loss_sum / len(train_dataset)
        train_regular_losses.append(train_regular_loss)
        train_trainable_losses.append(train_trainable_loss)
        print(f'Train loss: [regular: {train_regular_loss} trainable: {train_trainable_loss}]')
        if wandb_instance is not None:
            wandb_instance.log({
                'train': {
                    'regular_loss': train_regular_loss,
                    'trainable_loss': train_trainable_loss,
                },
                'lr': scheduler.get_last_lr()[0],
            }, step=epoch+1)

        scheduler.step()

        if (epoch + 1) % eval_every == 0:
        # eval
            model.eval()
            with torch.no_grad():
                regular_loss_sum = 0.
                trainable_loss_sum = 0.
                n_good = 0
                n_all = 0
                for input, target in test_dataloader:
                    input, target = input.to(device), target.to(device)
                    pred, feat = model(input)
                    
                    regular_loss = criterion(pred, target)
                    trainable_loss = trainable_criterion(feat, target) * training_kwargs['weight_criterion']
                    
                    classes = torch.argmax(pred, dim=1).cpu().numpy()
                    n_good += sum(classes == target.cpu().numpy())
                    n_all += len(classes)
                    
                    regular_loss_sum += regular_loss.item()
                    trainable_loss_sum += trainable_loss.item()

                val_regular_loss = regular_loss_sum / len(test_dataset)
                val_trainable_loss = trainable_loss_sum / len(test_dataset)
                test_regular_losses.append(val_regular_loss)
                test_trainable_losses.append(val_trainable_loss)

                print(f'Val : [regular_loss: {val_regular_loss}, trainable_loss: {val_trainable_loss}, accuracy: {n_good / n_all}]')
                if wandb_instance is not None:
                    wandb_instance.log({
                        'val': {
                            'regular_loss': val_regular_loss,
                            'trainable_loss': val_trainable_loss,
                            'accuracy': n_good / n_all,
                        },
                    }, step=epoch+1)

    return np.array(train_regular_losses), np.array(test_regular_losses), np.array(train_trainable_losses), np.array(test_trainable_losses), model