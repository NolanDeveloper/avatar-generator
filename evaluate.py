import cv2
import numpy as np
import torch
import torch._utils
import time
import os
import sys
import argparse
import pickle
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import get_network
from data import get_loader
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
from utils.metrics import MultiThresholdMeasures

def str2bool(s):
    return s.lower() in ('t', 'true', 1)

def evaluate(image):
#if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', help='path to ckpt file',type=str,
            default='./models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth')
    parser.add_argument('--dataset', type=str, default='figaro',
            help='Name of dataset you want to use default is "figaro"')
    parser.add_argument('--data_dir', help='path to Figaro1k folder', type=str, default='./data/Figaro1k')
    parser.add_argument('--networks', help='name of neural network', type=str, default='pspnet_resnet101')
    parser.add_argument('--save_dir', default='./overlay',
            help='path to save overlay images, default=None and do not save images in this case')
    parser.add_argument('--use_gpu', type=str2bool, default=False,
            help='True if using gpu during inference')

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    img_dir = os.path.join(data_dir, 'Original', 'Testing')
    network = args.networks.lower()
    save_dir = args.save_dir
    device = 'cuda' if args.use_gpu else 'cpu'

    assert os.path.exists(ckpt_dir)
    assert os.path.exists(data_dir)
    assert os.path.exists(os.path.split(save_dir)[0])

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # prepare network with trained parameters
    # net = get_network(network).to(device)
    # state = torch.load(ckpt_dir,  map_location=torch.device('cpu'))
    # net.load_state_dict(state['weight'])

    net = pickle.load(open('hair.dat', 'rb'))

    # this is the default setting for train_verbose.py
    # test_joint_transforms = jnt_trnsf.Compose([
    #     jnt_trnsf.Safe32Padding()
    # ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # transforms only on mask
    # mask_transforms = std_trnsf.Compose([
    #     std_trnsf.ToTensor()
    #     ])

    # test_loader = get_loader(dataset=args.dataset,
    #                          data_dir=data_dir,
    #                          train=False,
    #                          joint_transforms=test_joint_transforms,
    #                          image_transforms=test_image_transforms,
    #                          mask_transforms=mask_transforms,
    #                          batch_size=1,
    #                          shuffle=False,
    #                          num_workers=4)

    # prepare measurements
    metric = MultiThresholdMeasures()
    metric.reset()
    durations = list()

    # prepare images
    with torch.no_grad():
        topil_tf = std_trnsf.ToPILImage()
        data = jnt_trnsf.Safe32Padding()(topil_tf(image))
        data = test_image_transforms(data)
        data = torch.stack([data.to(device)], 0)
        net.eval()
        logit = net(data)
        pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
        mh, mw = data.size(2), data.size(3)
        mask = pred >= 0.5

        # mask_n = np.zeros((mh, mw, 3))
        # mask_n[:, :, 0] = 255
        # mask_n[:, :, 0] *= mask
        # mask_n = cv2.cvtColor(mask_n,
        #                          cv2.COLOR_GRAY2BGR)  # change mask to a 3 channel image
        # mask_out = cv2.subtract(mask_n, imgs)
        # mask_out = cv2.subtract(mask_n, mask_out)

    #return mask_out
            # cv2.imshow('Mask', mask_n)
      #  cv2.waitKey(0)
        return mask



