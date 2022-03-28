import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sewar as swr


def convert_tensor_into_image(tensor):
    img = tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = img * std + mean
    np.clip(img, 0, 1)
    return img


def show_image_from_tensor(tensor, title="No title", figsize=(5, 5)):
    img = convert_tensor_into_image(tensor)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)


def weights_init(m):
    name = m.__class__.__name__

    if(name.find("Conv") > -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # ~N(mean=0.0, std=0.02)
    elif(name.find("BatchNorm") > -1):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def all_measures_from_tensors(tensor1, tensor2):
    img1 = convert_tensor_into_image(tensor1)
    img1 = (img1 * 255).astype(int)

    img2 = convert_tensor_into_image(tensor2)
    img2 = (img2 * 255).astype(int)

    dict = {}
    dict['RMSE'] = swr.rmse(img1, img2)
    dict['PSNR'] = swr.psnr(img1, img2)
    dict['SSIMx'] = swr.ssim(img1, img2)[0]
    dict['SSIMy'] = swr.ssim(img1, img2)[1]
    dict['UQI'] = swr.uqi(img1, img2)
    dict['VIF'] = swr.vifp(img1, img2)
    dict['ERGAS'] = swr.ergas(img1, img2)
    dict['SCC'] = swr.scc(img1, img2)

    return dict
