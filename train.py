from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from hyper_parameters import NGPU, DEVICE, BATCH_SIZE, LEARNING_RATE, BETA1, BETA2, NUM_EPOCHS, L1_LAMBDA, L2_LAMBDA, P_LAMBDA
from data_load import dataloader_train
from functions import weights_init
from networks import Generator, Discriminator

print('--------------------------------------------------------')
print('INITIALIZATION')
print('--------------------------------------------------------')
print('DEVICE :',DEVICE)
print('NGPU :',NGPU)
print('--------------------------------------------------------')
print('HYPERPARAMETERS')
print('--------------------------------------------------------')
print('BATCH_SIZE :',BATCH_SIZE)
print('LEARNING_RATE :',LEARNING_RATE)
print('BETA1 :',BETA1)
print('BETA2 :',BETA2)
print('L1_LAMBDA :',L1_LAMBDA)
print('L2_LAMBDA :',L2_LAMBDA)
print('P_LAMBDA :',P_LAMBDA)

# Loading data

images, _ = next(iter(dataloader_train))

# Initialization of the generator

model_G = Generator(ngpu=NGPU)

if(DEVICE == "cuda" and NGPU > 1):
    model_G = nn.DataParallel(model_G, list(range(NGPU)))

model_G.apply(weights_init)
model_G.to(DEVICE)

# Initialization of the discriminator

model_D = Discriminator(ngpu=NGPU)

if(DEVICE == "cuda" and NGPU > 1):
    model_D = torch.DataParallel(model_D, list(range(NGPU)))

model_D.apply(weights_init)
model_D.to(DEVICE)

out1 = model_D(torch.cat([images[:, :, :, :256].to(
    DEVICE), images[:, :, :, 256:].to(DEVICE)], dim=1)).to(DEVICE)
out2 = torch.ones(size=out1.shape, dtype=torch.float, device=DEVICE)

# Initialization of the loss network

if P_LAMBDA > 0:
    model_P = torchvision.models.vgg19(pretrained=True)
    model_P.to(DEVICE)

# Loss Definition

GAN_Loss = nn.BCELoss()
L1_Loss = nn.L1Loss(reduction='mean')
L2_Loss = nn.MSELoss(reduction='mean')


def D_Loss(outputs, labels):
    # divide the loss by 2 -> slow down D
    loss = 0.5*GAN_Loss(outputs, labels)
    return loss


def G_Loss(outputs, labels, gens, targets):
    loss = GAN_Loss(outputs, labels)
    if L1_LAMBDA > 0:
        loss += L1_LAMBDA * L1_Loss(gens, targets)
    if L2_LAMBDA > 0:
        loss += L2_LAMBDA * L2_Loss(gens, targets)
    if P_LAMBDA > 0:
        feature_gens = model_P(gens)
        feature_targets = model_P(targets)
        loss += P_LAMBDA * L2_Loss(feature_gens, feature_targets)
    return loss


# Optimizer Definition


optimizerD = optim.Adam(model_D.parameters(),
                        lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerG = optim.Adam(model_G.parameters(),
                        lr=LEARNING_RATE, betas=(BETA1, BETA2))

# Create instance of Tensorboard

writer = SummaryWriter()

# Training

print('--------------------------------------------------------')
print('TRAINING')
print('--------------------------------------------------------')

for epoch in range(NUM_EPOCHS+1):
    start = time.time()

    for images, _ in iter(dataloader_train):
        # ========= Train Discriminator ===========
        # Train on real data
        # Maximize log(D(x,y)) <- maximize D(x,y)
        model_D.zero_grad()

        inputs = images[:, :, :, :256].to(DEVICE)  # input image data
        targets = images[:, :, :, 256:].to(DEVICE)  # real targets data

        real_data = torch.cat([inputs, targets], dim=1).to(DEVICE)
        outputs = model_D(real_data)  # label "real" data
        labels = torch.ones(size=outputs.shape,
                            dtype=torch.float, device=DEVICE)

        lossD_real = D_Loss(outputs, labels)
        lossD_real.backward()

        # Train on fake data
        # Maximize log(1-D(x,G(x))) <- minimize D(x,G(x))
        gens = model_G(inputs).detach()

        fake_data = torch.cat([inputs, gens], dim=1)  # generated image data
        outputs = model_D(fake_data)
        # label "fake" data
        labels = torch.zeros(size=outputs.shape,
                             dtype=torch.float, device=DEVICE)

        lossD_fake = D_Loss(outputs, labels)
        lossD_fake.backward()

        optimizerD.step()

        # ========= Train Generator x2 times ============
        # maximize log(D(x, G(x)))
        for i in range(2):
            model_G.zero_grad()

            gens = model_G(inputs)

            # concatenated generated data
            gen_data = torch.cat([inputs, gens], dim=1)
            outputs = model_D(gen_data)
            labels = torch.ones(size=outputs.shape,
                                dtype=torch.float, device=DEVICE)

            lossG = G_Loss(outputs, labels, gens, targets)
            lossG.backward()
            optimizerG.step()

        # ========= Train Generator x2 times ============
        # maximize log(D(x, G(x)))

    # Visualization with TensorBoard

    writer.add_scalar('Loss/Discriminator_loss_on_real_data',
                      lossD_real.item(), epoch)
    writer.add_scalar('Loss/Discriminator_loss_on_fake_data',
                      lossD_fake.item(), epoch)
    writer.add_scalar('Loss/Generator_loss', lossG.item(), epoch)
    writer.add_image('Images', torch.cat(
        (inputs, gens, targets), dim=3).detach().cpu()[0], epoch)

    # Saving trained

    if(epoch % 20 == 0):
        torch.save(
            model_G, "./trained_networks/old/generator_epoch_" + str(epoch) + ".pth")
        torch.save(
            model_D, "./trained_networks/old/discriminator_epoch_" + str(epoch) + ".pth")

    torch.save(model_G, "./trained_networks/generator_last.pth")
    torch.save(model_D, "./trained_networks/discriminator_last.pth")

    print(f"TRAINING EPOCH {epoch} /",NUM_EPOCHS,", DURATION :",
          round(time.time()-start, 1),'sec')
    


writer.close()

print('--------------------------------------------------------')
print("TRAINING DONE !")
print('--------------------------------------------------------')