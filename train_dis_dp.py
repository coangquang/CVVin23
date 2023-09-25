import os
import time
import numpy as np
from skimage import io
import time
from tqdm import tqdm


import argparse
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

from torch.optim import AdamW
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from utils import mask_image_list
from visualization import plot_loss_vs_epochs

from model.isnet_dpconv import *
def get_args():
    parser = argparse.ArgumentParser('Sailent object detection')
    parser.add_argument('--batch_size', type=int, default = 4, help = 'The number of sample per batch among all devices')
    parser.add_argument('--num_epochs', type=int, default = 100)
    parser.add_argument('--saved_path', type=str, default = 'logs/ISNet')
    parser.add_argument("--weight_GT", type=str, default = None, help = 'Weights of ground-truth encoder')
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--resume', type=str, default = None, help  ='path for weights to continue training')
    args = parser.parse_args()
    return args

def train(opt):
    tra_img_name_list, tra_lbl_name_list = mask_image_list('DUTS-TR')
    salobj_dataset = SalObjDataset( img_name_list=tra_img_name_list,
                                   lbl_name_list=tra_lbl_name_list,
                                   transform = transforms.Compose([
                                                        RescaleT(1024),
                                                        RandomCrop(920),
                                                        ToTensorLab(flag=0)]))

    img_name_list, lbl_name_list = mask_image_list('DUTS-TE')
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(1024),
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
    

    net1 = ISNetGTEncoder()   
    net2 = ISNetDIS(3, 1)
    if torch.cuda.is_available():
        net1.cuda()
        net2.cuda()

    print(summary(net2, (3, 1024, 1024)))
    if opt.weight_GT is not None:
        checkpoint = torch.load(opt.weight_GT)
        net1.load_state_dict(checkpoint['model_state_dict'])


    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)
        net2.load_state_dict(checkpoint['model_state_dict'])
    optimizer = AdamW(net2.parameters(), lr = opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, 
                      weight_decay=0.01)

    epochs = opt.num_epochs

    # Initialize train and validation losses lists
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Set model in train mode
        net2.train()
        net1.eval()
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
            ds, dfs = net2(inputs_v)
            _, fs = net1(labels_v) ## extract the gt encodings
            loss2, loss1 = net2.compute_loss_kl(ds, labels_v, dfs, fs, mode='MSE')

            loss1.backward()
            optimizer.step()

            # total loss
            loss_of_epoch += loss1.item()

            # del temporary outputs and loss
            del ds, loss2, loss1, fs, dfs

        loss_of_epoch /= len(salobj_dataset)
        train_losses.append(loss_of_epoch)

        # Set model in evaluation mode
        net2.eval() 
        net1.eval()
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
                

                ds,_ = net2(inputs_v)
                loss0, loss1 = net.compute_loss(ds, labels_v)
                # Find the total loss
                loss_of_epoch += loss0.item()
                # del temporary outputs and loss
                del ds, loss0, loss1
        
        # Print each epoch's time and train/val loss 
        loss_of_epoch /= len(test_salobj_dataset)
        val_losses.append(loss_of_epoch)

        print("------- Epoch ", epoch+1 ," -------"", Training Loss:", train_losses[-1], ", Validation Loss:", val_losses[-1], "\n")
        plot_loss_vs_epochs(train_losses, val_losses)

        if (epoch + 1) % 1 == 0:
            saved_path = opt.saved_path + f'/{epoch + 1}.pt'
            torch.save({
                'model_state_dict': net2.state_dict(),
                'epoch': epoch,
                }, saved_path)
if __name__ == '__main__':
    opt = get_args()
    print('\n')
    print("Describe model: ",  opt)
    print("\n")
    train(opt)
