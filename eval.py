from numpy import average
import torch
import torchvision
import matplotlib.pyplot as plt
import time
import numpy as np

from hyper_parameters import DEVICE
from data_load import dataloader_val
from functions import show_image_from_tensor, all_measures_from_tensors

model_G = torch.load("./trained_networks/generator_last_GAN_L1_200.pth").to(DEVICE)

test_imgs, _ = next(iter(dataloader_val))

satellite = test_imgs[:, :, :, :256].to(DEVICE)
maps = test_imgs[:, :, :, 256:].to(DEVICE)

gen = model_G(satellite)

satellite = satellite.detach().cpu()
gen = gen.detach().cpu()
maps = maps.detach().cpu()

show_image_from_tensor(torchvision.utils.make_grid(satellite, padding=10),
                       title="Satellite", figsize=(50, 50))
show_image_from_tensor(torchvision.utils.make_grid(gen, padding=10),
                       title="Generated", figsize=(50, 50))
show_image_from_tensor(torchvision.utils.make_grid(maps, padding=10),
                       title="Expected Output", figsize=(50, 50))
show_image_from_tensor(torch.cat(
        (satellite, gen, maps), dim=3).detach().cpu()[0],
                       title="", figsize=(50, 50))
plt.show()

print('--------------------------------------------------------')
print('MEASURES CALCULATION')
print('--------------------------------------------------------')

n = satellite.shape[0]
rep = int(1098 / n)

list_dict = {}

for r in range(rep):
    start = time.time()

    test_imgs, _ = next(iter(dataloader_val))

    satellite = test_imgs[:, :, :, :256].to(DEVICE)
    maps = test_imgs[:, :, :, 256:].to(DEVICE)

    gen = model_G(satellite)

    satellite = satellite.detach().cpu()
    gen = gen.detach().cpu()
    maps = maps.detach().cpu()

    for i in range(n):
        dict = all_measures_from_tensors(maps[i], gen[i])
        for key in dict.keys():
            if key in list_dict:
                list_dict[key].append(dict[key])
            else:
                list_dict[key] = [dict[key]]

    print('BATCH ',r+1,'/',rep,", DURATION :",
          round(time.time()-start, 1),'sec')

mean_dict = {}
for key in list_dict.keys():
    mean_dict[key] = np.mean(list_dict[key])

print('--------------------------------------------------------')
print('QUANTITATIVE RESULTS BETWEEN TARGET AND GENERATED IMAGES')
print('--------------------------------------------------------')
print("* RMSE: {}".format(mean_dict['RMSE']))
print("* PSNR: {}".format(mean_dict['PSNR']))
print("* SSIMx: {}".format(mean_dict['SSIMx']))
print("* SSIMy: {}".format(mean_dict['SSIMy']))
print("* UQI: {}".format(mean_dict['UQI']))
print("* VIF: {}".format(mean_dict['VIF']))
print("* ERGAS: {}".format(mean_dict['ERGAS']))
print("* SCC: {}".format(mean_dict['SCC']))
print('--------------------------------------------------------')
