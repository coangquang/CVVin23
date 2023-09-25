from model.u2net_dpconv import U2NET_DP
from model.u2net import U2NET
from model.isnet import ISNetDIS
from model.isnet_dpconv import ISNetDIS_DP

import torch
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from utils import mask_image_list
from torchvision import transforms, utils
import argparse
from tqdm import tqdm
from metric.metric import mae, wfm, sm

def get_args():
    parser = argparse.ArgumentParser('Sailent object detection')
    parser.add_argument('--model', type=str, default = 'u2net', help = 'name model')
    parser.add_argument("--checkpoint", type=str, default = None)
    args = parser.parse_args()
    return args

def eval(opt):
    img_name_list, lbl_name_list = mask_image_list('DUTS-TE')
    if opt.model == 'u2net':
        net = U2NET(3, 1)
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    elif opt.model =='u2net-dp':
        net = U2NET_DP(3, 1)
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)]))

    elif opt.model == 'dis':
        net = ISNetDIS(3, 1)
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(1024),
                                                                      ToTensorLab(flag=0)]))
    else:
        net = ISNetDIS_DP(3, 1)
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        transform=transforms.Compose([RescaleT(1024),
                                                                      ToTensorLab(flag=0)]))
    if opt.checkpoint is not None:
        checkpoint = torch.load(opt.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        net.cuda()
    total_mae = 0
    total_wfm = 0
    total_sm = 0
    for i in tqdm(range(len(test_salobj_dataset))):
        with torch.no_grad():
            net.eval()
            input = test_salobj_dataset[i]['image']
            label = test_salobj_dataset[i]['label']
            if torch.cuda.is_available():
                inputs_v =  input.cuda()
                labels_v = label.cuda()
            else:
                inputs_v =  input
                labels_v = label 
            if opt.model == 'u2net':
                prediction, _, _, _, _, _, _ = net(inputs_v.unsqueeze(0).type(torch.cuda.FloatTensor))
            elif opt.model == 'u2net-dp':
                prediction, _, _, _, _, _, _ = net(inputs_v.unsqueeze(0).type(torch.cuda.FloatTensor))
            else:
                result, _ = net(inputs_v.unsqueeze(0).type(torch.cuda.FloatTensor))
                prediction = result[0]

            total_mae += mae(prediction.squeeze().cpu().numpy(), 
                             labels_v.squeeze().cpu().numpy())
            total_wfm += wfm(prediction.squeeze().cpu().numpy() * 255, 
                              labels_v.squeeze().cpu().numpy() * 255)
            total_sm += sm(prediction.squeeze().cpu().numpy(), 
                               labels_v.squeeze().cpu().numpy())
    avg_mae = total_mae / len(test_salobj_dataset)
    avg_wfm = total_wfm / len(test_salobj_dataset)
    avg_sm = total_sm / len(test_salobj_dataset)
    print("MAE score: ", avg_mae ,", Weighted F-measure: ", avg_wfm, ", S-measure: ", avg_sm)
if __name__ == '__main__':
    opt = get_args()
    print('\n')
    print("Describe model: ",  opt)
    print("\n")
    eval(opt)