import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean
from dataset import VioDB
from config import Config
from spatial_transforms import Compose, ToTensor, Normalize
from spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from temporal_transforms import CenterCrop, RandomCrop
from target_transforms import Label, Video
from utils import Log

def main(config):
    # Load model
    if config.model == 'c3d':
        model, params = VioNet_C3D(config)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config)
    else:
        model, params = VioNet_densenet_lean(config)

    # Dataset and DataLoader setup
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration
    cv = config.num_cv

    # Define transforms and loaders for training and validation
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose([crop_method, GroupRandomHorizontalFlip(), ToTensor(), norm])
    temporal_transform = RandomCrop(size=sample_duration, stride=stride)
    target_transform = Label()
    train_batch = config.train_batch
    train_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                       '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                       spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True, num_workers=4, pin_memory=True)

    # Validation set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    val_batch = config.val_batch
    val_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                     '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                     spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False, num_workers=4, pin_memory=True)

    # Make directories for saving logs and checkpoints
    if not os.path.exists('./pth'):
        os.mkdir('./pth')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # Logging
    batch_log = Log('./log/{}_fps{}_{}_batch{}.log'.format(config.model, sample_duration, dataset, cv),
                    ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log('./log/{}_fps{}_{}_epoch{}.log'.format(config.model, sample_duration, dataset, cv),
                    ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log('./log/{}_fps{}_{}_val{}.log'.format(config.model, sample_duration, dataset, cv),
                  ['epoch', 'loss', 'acc'])

    # Prepare criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=params,
                                lr=config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=config.factor, min_lr=config.min_lr)

    # Load checkpoint if exists
    start_epoch = 0
    checkpoint_path = './pth/latest_checkpoint.pth'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        acc_baseline = checkpoint.get('val_acc', config.acc_baseline)
        loss_baseline = checkpoint.get('val_loss', 1)
        print(f"Checkpoint loaded: Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config.num_epoch):
        train(epoch, train_loader, model, criterion, optimizer, device, batch_log, epoch_log)
        val_loss, val_acc = val(epoch, val_loader, model, criterion, device, val_log)
        scheduler.step(val_loss)

        # Save checkpoint
        if val_acc > acc_baseline or (val_acc >= acc_baseline and val_loss < loss_baseline):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            acc_baseline = val_acc
            loss_baseline = val_loss
            print(f"Checkpoint saved at epoch {epoch}")

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config('densenet_lean', 'hockey', device=device, num_epoch=150, acc_baseline=0.92, ft_begin_idx=0)

    configs = {
        'hockey': {'lr': 1e-2, 'batch_size': 32},
        'movie': {'lr': 1e-3, 'batch_size': 16},
        'vif': {'lr': 1e-3, 'batch_size': 16}
    }

    for dataset in ['hockey', 'movie', 'vif']:
        config.dataset = dataset
        config.train_batch = configs[dataset]['batch_size']
        config.val_batch = configs[dataset]['batch_size']
        config.learning_rate = configs[dataset]['lr']
        config.num_cv = 1
        main(config)
