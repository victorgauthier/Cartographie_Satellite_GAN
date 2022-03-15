import torch
import torchvision

from hyper_parameters import device
from data_load import dataloader_val
from functions import weights_init, show_image

model_G = torch.load("./Generator.pth")
model_G.apply(weights_init)
test_imgs,_ = next(iter(dataloader_val))

satellite = test_imgs[:,:,:,:256].to(device)
maps = test_imgs[:,:,:,256:].to(device)

gen = model_G(satellite)
#gen = gen[0]

satellite = satellite.detach().cpu()
gen = gen.detach().cpu()
maps = maps.detach().cpu()

show_image(torchvision.utils.make_grid(satellite, padding=10), title="Satellite", figsize=(50,50))
show_image(torchvision.utils.make_grid(gen, padding=10), title="Generated", figsize=(50,50))
show_image(torchvision.utils.make_grid(maps, padding=10), title="Expected Output", figsize=(50,50))