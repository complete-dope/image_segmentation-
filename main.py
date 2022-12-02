from data_loader_cache import normalize, im_reader, im_preprocess
from models import *
from io import BytesIO
import requests
import io
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")


# project imports

device = 'cpu'


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose(
    [GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(im_path, hypar):
    if im_path.startswith("http"):
        im_path = BytesIO(requests.get(im_path).content)

    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    # make a batch of image, shape
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"] != ""):
        net.load_state_dict(torch.load(
            hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(
        device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    # B x 1 x H x W    # we want the first one which is the most accurate prediction
    pred_val = ds_val[0][0, :, :, :]

    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(
        pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi)  # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    # it is the mask we need
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8), pred_val


hypar = {}  # paramters for inferencing

hypar["model_path"] = "./saved_models"  # load trained weights from this path
# name of the to-be-loaded weights
hypar["restore_model"] = "isnet-general-use.pth"
# indicate if activate intermediate feature supervision
hypar["interm_sup"] = False

# choose floating point accuracy --
# indicates "half" or "full" accuracy of float number
hypar["model_digit"] = "full"
hypar["seed"] = 0

# cached input spatial resolution, can be configured into different size
hypar["cache_size"] = [1024, 1024]

# data augmentation parameters ---
# mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["input_size"] = [1024, 1024]
# random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
hypar["crop_size"] = [1024, 1024]

hypar["model"] = ISNetDIS()


net = build_model(hypar, device)


def input_output_image(image_path , output_path):
    byteImgIO = io.BytesIO()
    byteImg = Image.open(image_path)
    byteImg.save(byteImgIO, "JPEG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()

    image_tensor, orig_size = load_image(image_path, hypar) 
    mask , masked = predict(net,image_tensor,orig_size, hypar, device)

    inv_mask = cv2.subtract(255, mask) 
    a,b = inv_mask.shape[0] , inv_mask.shape[1] 
    image_imread = plt.imread(image_path)

    image = cv2.cvtColor(image_imread, cv2.COLOR_BGR2GRAY)
    masked = masked.detach().numpy()
    neww_img = image*masked

    new_new_img = neww_img
    for i in range(0 , a):
        for j in range(0 , b):
            if inv_mask[i][j] >= 150:
                new_new_img[i][j] = 255

    im = Image.fromarray(new_new_img)
    im = im.convert("L")
    im.save(output_path) 


def output_colored_img(input_path , output_path):
    byteImgIO = io.BytesIO()
    byteImg = Image.open(input_path)
    byteImg.save(byteImgIO, "JPEG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()

    image_imread = plt.imread(input_path)
    image_tensor, orig_size = load_image(input_path, hypar) 
    mask , masked = predict(net,image_tensor,orig_size, hypar, device) # this is a 2d image


    inv_mask = cv2.subtract(255, mask) 
    inv_mask_3d = cv2.cvtColor(inv_mask , cv2.COLOR_GRAY2RGB) #make it 3d
    a,b,c = inv_mask_3d.shape

    copied_mask_2d = mask
    copied_mask_3d = copied_mask_2d
    copied_invmask_2d = inv_mask
    copied_invmask_3d = copied_invmask_2d
    # make then rgb images
    copied_mask_3d = cv2.cvtColor(copied_mask_2d , cv2.COLOR_GRAY2RGB)
    # print('shape would be changed for copied mask 2d also ' , copied_mask_2d.shape)

    copied_invmask_3d = cv2.cvtColor(copied_invmask_2d , cv2.COLOR_GRAY2RGB)
    # print('shape would be changed for copied invmask 2d also ' , copied_mask_2d.shape)
    a,b,c = copied_mask_3d.shape 

    my_last_semi_final_img = np.ndarray(shape = (a,b,c) , dtype = np.uint8)
    # car in same color with black bg
    for i in range(0 , a):
        for j in range(0 , b):
            for k in range(0,c):
                if(copied_mask_3d[i][j][k] >= 240):
                    my_last_semi_final_img[i][j][k] = image_imread[i][j][k]

    # print('done')

    #expecting a car with black bg 
    # plt.imshow(my_last_semi_final_img)
    # plt.show();

    #make copies 
    copied_black_bg_img = my_last_semi_final_img
    copied_2_black_bg_img =copied_black_bg_img
    my_last_final_img = copied_black_bg_img
    # car in same color with white bg

    # plt.show()

    for i in range(0 , a):
        for j in range(0 , b):
            for k in range(0,c):
                if(copied_invmask_3d[i][j][k] >= 238): #car in black rest in white
                    my_last_final_img[i][j][k] = 255

    # print('done')
    plt.imshow(my_last_final_img)

    my_final_img_copy = my_last_final_img
    my_final_2_img_copy = my_final_img_copy
    
    r,g,b = cv2.split(my_final_2_img_copy)

    merged_for_opencv = cv2.merge([b,g,r])

    cv2.imwrite(output_path , merged_for_opencv)

output_colored_img(); 