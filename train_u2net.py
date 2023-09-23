import argparse
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
from torch.optim import AdamW
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from utils import mask_image_list
from metric.loss import muti_bce_loss_fusion
from visualization import plot_loss_vs_epochs

from model.u2net import U2NET

def get_args():
    parser = argparse.ArgumentParser('Sailent object detection')
    parser.add_argument('--model', type=str, default = 'u2net', help = 'name model')
    parser.add_argument('--batch_size', type=int, default = 4, help = 'The number of sample per batch among all devices')
    parser.add_argument('--num_epochs', type=int, default = 1000)
    parser.add_argument('--saved_path', type=str, default = 'logs')
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--resume', type=str, default = None, help  ='path for weights to continue training')
    args = parser.parse_args()
    return args

def train(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available()else 'cpu')

    tra_img_name_list, tra_lbl_name_list = mask_image_list('DUTS-TR')
    salobj_dataset = SalObjDataset( img_name_list=tra_img_name_list,
                                   lbl_name_list=tra_lbl_name_list,
                                   transform = transforms.Compose([
                                                        RescaleT(320),
                                                        RandomCrop(288),
                                                        ToTensorLab(flag=0)]))

    img_name_list, lbl_name_list = mask_image_list('DUTS-TE')
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    
    salobj_dataloader = DataLoader(salobj_dataset, 
                                   batch_size=opt.batch_size, 
                                   shuffle=True, 
                                   num_workers=1)
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    if (opt.model == 'u2net'):
        net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.cuda()
    print(summary(net, (3, 320, 320)))
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)
        net.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = AdamW(net.parameters(), lr = opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, 
                      weight_decay=0.01)

    epochs = opt.num_epochs

    # Initialize train and validation losses lists
    train_losses = []
    val_losses = []


    for epoch in range(epochs):
        # Set model in train mode
        net.train()
        
        loss_of_epoch = 0

        print("----------Train----------")
        for batch_idx, data  in enumerate(tqdm(salobj_dataloader)):
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss0, loss1 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss1.backward()
            optimizer.step()

            # total loss
            loss_of_epoch += loss1
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss0, loss1
        loss_of_epoch /= len(salobj_dataset)
        train_losses.append(loss_of_epoch.cpu())
        
        # Set model in evaluation mode
        net.eval() 
        print("----------Evaluate----------")
        loss_of_epoch = 0
        for batch_idx, data in enumerate(tqdm(test_salobj_dataloader)): 
            with torch.no_grad():
                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss0, loss1 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                # Find the total loss
                loss_of_epoch += loss0

                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss0, loss1
        
        # Print each epoch's time and train/val loss 
        loss_of_epoch /= len(test_salobj_dataset)
        val_losses.append(loss_of_epoch.cpu())
        
        print("------- Epoch ", epoch+1 ," -------"", Training Loss:", train_losses[-1], ", Validation Loss:", val_losses[-1], "\n")
        plot_loss_vs_epochs(train_losses, val_losses)
        
        saved_path = opt.saved_path + f'/{epoch}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            }, saved_path)
if __name__ == '__main__':
    opt = get_args()
    print('\n')
    print("Describe model: ",  opt)
    print("\n")
    train(opt)
