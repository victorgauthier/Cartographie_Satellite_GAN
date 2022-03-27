from numpy import average
import torch
import torchvision
import matplotlib.pyplot as plt

from hyper_parameters import DEVICE
from data_load import dataloader_val, dataloader_val_big
from functions import show_image_from_tensor, all_measures_from_tensors

model_G = torch.load("./trained_networks/generator_last.pth")

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
plt.show()

test_imgs, _ = next(iter(dataloader_val_big))

satellite = test_imgs[:, :, :, :256].to(DEVICE)
maps = test_imgs[:, :, :, 256:].to(DEVICE)

gen = model_G(satellite)

satellite = satellite.detach().cpu()
gen = gen.detach().cpu()
maps = maps.detach().cpu()

average_dict = {}
n = len(satellite.shape[0])
for i in range(n):
    dict = all_measures_from_tensors(maps[i], gen[i])
    for key in dict.keys():
        if key in average_dict:
            average_dict[key] += dict[key]
        else:
            average_dict[key] = dict[key]
for key in average_dict.keys():
    average_dict[key] = average_dict[key] / n

print('--------------------------------------------------------')
print('QUANTITATIVE RESULTS BETWEEN TARGET AND GENERATED IMAGES')
print('--------------------------------------------------------')
print("* RMSE: {}".format(average_dict['RMSE']))
print("* PSNR: {}".format(average_dict['PSNR']))
print("* SSIM: {}".format(average_dict['SSIM']))
print("* UQI: {}".format(average_dict['UQI']))
print("* VIF: {}".format(average_dict['VIF']))
print("* ERGAS: {}".format(average_dict['ERGAS']))
print("* SCC: {}".format(average_dict['SCC']))
print('--------------------------------------------------------')
