from charades_dataset import Charades as Dataset
from pytorch_i3d import InceptionI3D
import numpy as np
import videotransforms
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.data.dataloader
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='',
        train_split='',
        batch_size=8*5,
        save_model=''):
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip()
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224)
    ])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    if mode == 'flow':
        i3d = InceptionI3D(400, in_channels=2)
        i3d.load_state_dict(torch.load(''))
    else:
        i3d = InceptionI3D(400, in_channels=3)
        i3d.load_state_dict(torch.load(''))
    i3d.replace_logits(157)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.0001)
    lr_schedule = optim.lr_scheduler.MultiplicativeLR(optimizer, [300, 1000])

    num_steps_per_update = 4
    steps = 4
    while steps < max_steps:
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            for data in dataloader[phase]:
                num_iter += 1
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                per_frame_logits = F.upsample(
                    per_frame_logits, t, mode='linear')

                loc_loss = F.binary_cross_entropy_with_logits(
                    per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]

                cls_loss = F.binary_cross_entropy_with_logits(
                    torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_schedule.step()
                    if steps % 10 == 0:
                        print('{} Loc Loss: {: .4f} Cls Loss: {: .4f} Tot Loss: {: .4f}'.format(
                            phase, tot_loc_loss / (10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update), tot_loss / 10))
                        torch.save(i3d.module.state_dict(),
                                   save_model + str(steps).zfill(6) + '.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.0

            if phase == 'val':
                print('{} Loc Loss: {: .4f} Cls Loss: {: .4f} Tot Loss: {: .4f}'.format(
                    phase, tot_loc_loss / num_iter, tot_cls_loss / num_iter, (tot_loss * num_steps_per_update) / num_iter))


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, save_model=args.save_model)
