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

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_losses = []

    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        prediction = model(x)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return train_losses


def eval_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            prediction = model(x)
            loss = criterion(prediction, target)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args, criterion, device, wandb_instance=None):
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_args['step_size'], gamma=train_args['gamma'])

    train_losses = []
    test_losses = [eval_loss(model, test_loader, criterion, device)]

    print('initial loss {}'.format(test_losses[-1]))
    if wandb_instance is not None:
        wandb_instance.log({
            'val': {
                'loss': test_losses[-1],
            },
        }, step=0)

    for epoch in range(epochs):
        print(f'epoch {epoch} started')

        model.train()
        train_loss = train(model, train_loader, optimizer, criterion, device)        
        test_loss = eval_loss(model, test_loader, criterion, device)

        scheduler.step()

        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        print('train loss: {}, test_loss: {}'.format(np.mean(train_loss), 
                                                     test_losses[-1]))
        if wandb_instance is not None:
            wandb_instance.log({
                'val': {
                    'loss': test_losses[-1],
                },
                'train': {
                    'loss': np.mean(train_loss),
                },
                'lr': scheduler.get_last_lr()[0],
            }, step=epoch+1)

    return train_losses, test_losses


def train_model(train_dataset, test_dataset, model, criterion, device, train_dataloader_kwargs, test_dataloader_kwargs, training_kwargs, wandb_instance=None):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    model: nn.Model item, should contain function loss and accept
    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - trained model
    """
    model.to(device)

    train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **test_dataloader_kwargs)

    train_loss, test_loss = train_epochs(model, train_dataloader, test_dataloader, training_kwargs, criterion, device, wandb_instance=wandb_instance)

    return np.array(train_loss), np.array(test_loss), model
    

import sys
import warnings

import numpy as np

import torch
from torch import optim as optim

from torchvision.transforms import functional as F

DEFAULT_IMAGE_SIZE = (256, 256)


def train_old(dataset, net=None, criterion=None, batch_size=8, lr=3e-4, epochs=20, device=None, wandb_instance=None):
    if device is not None:
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    stats_step = (len(dataset) // 10 // batch_size) + 1

    step = 0

    for epoch in range(epochs):
        if epoch == 0:
            # на первой эпохе учимся с малым lr, чтобы не сломать pretrain
            optimizer.lr = lr / 1000
        else:
            # дальше постепенно уменьшаем
            optimizer.lr = lr / 2**epoch

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, anno = data
            if device is not None:
                inputs = inputs.to(device)
                anno = anno.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, anno)
            if torch.isnan(loss).any():
                warnings.warn("nan loss! skip update")
                print(f"last loss: {loss}")
                break
            running_loss += loss
            if (i % stats_step == 0):
                print(f"epoch {epoch}|{i}; total loss:{running_loss / stats_step}")
                print(f"last losse: {loss}")

                if wandb_instance is not None:
                    wandb_instance.log({
                        'train': {
                            'loss': running_loss / stats_step,
                            'lr': optimizer.param_groups[0]['lr']
                        },
                    }, step=step)
                step += 1

                running_loss = 0.0
            loss.backward()
            optimizer.step()
    print('Finished Training')
    return net