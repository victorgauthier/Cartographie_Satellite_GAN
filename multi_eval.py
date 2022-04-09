from numpy import average
import torch
import torchvision
import matplotlib.pyplot as plt
import time
import numpy as np

from hyper_parameters import DEVICE
from data_load import dataloader_show
from functions import show_image_from_tensor,convert_tensor_into_image

path_list = ["GAN_100","L1_200","L2_200","GAN_L1_200","GAN_L2_200","GAN_P_100","GAN_L1_P_100"]

test_imgs, _ = next(iter(dataloader_show))
satellite = test_imgs[:, :, :, :256].to(DEVICE)
maps = test_imgs[:, :, :, 256:].to(DEVICE)
satellite = satellite.detach().cpu()
maps = maps.detach().cpu()

figure, axis = plt.subplots(2, 5)

img_satellite = convert_tensor_into_image(satellite[0])
axis[0, 0].imshow(img_satellite)
axis[0, 0].set_title("Satellite")
axis[0, 0].xaxis.set_visible(False) 
axis[0, 0].yaxis.set_visible(False) 

img_map = convert_tensor_into_image(maps[0])
axis[0, 1].imshow(img_map)
axis[0, 1].set_title("True map") 
axis[0, 1].xaxis.set_visible(False) 
axis[0, 1].yaxis.set_visible(False) 

c = 0
for path in path_list:

    model_G = torch.load("./trained_networks/old_"+path+"/generator_epoch_100.pth").to(DEVICE)

    satellite = test_imgs[:, :, :, :256].to(DEVICE)
    maps = test_imgs[:, :, :, 256:].to(DEVICE)

    gen = model_G(satellite)

    satellite = satellite.detach().cpu()
    gen = gen.detach().cpu()
    maps = maps.detach().cpu()

    img_generated = convert_tensor_into_image(gen[0])

    if 2+c >= 5:
        r = 1
    else:
        r = 0

    axis[r, (2 + c)%5].imshow(img_generated)
    axis[r, (2 + c)%5].set_title("("+str(c+1)+") "+path[:-4]) 
    axis[r, (2 + c)%5].xaxis.set_visible(False) 
    axis[r, (2 + c)%5].yaxis.set_visible(False) 

    c += 1

axis[1, 4].xaxis.set_visible(False) 
axis[1, 4].yaxis.set_visible(False)

plt.show()