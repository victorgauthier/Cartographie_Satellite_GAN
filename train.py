import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from hyper_parameters import ngpu, device, lr, beta1, beta2, NUM_EPOCHS, L1_lambda
from data_load import dataloader_train
from functions import weights_init
from networks import Generator, Discriminator

# Loading data

images,_ = next(iter(dataloader_train))

# Initialization of the generator

model_G = Generator(ngpu=1)

if(device == "cuda" and ngpu > 1):
    model_G = nn.DataParallel(model_G, list(range(ngpu)))
    
model_G.apply(weights_init)
model_G.to(device)

# Initialization of the discriminator

model_D = Discriminator(ngpu=1)

if(device == "cuda" and ngpu>1):
    model_D = torch.DataParallel(model_D, list(range(ngpu)))
    
model_D.apply(weights_init)
model_D.to(device)

out1 = model_D(torch.cat([images[:,:,:,:256].to(device), images[:,:,:,256:].to(device)], dim=1)).to(device)
out2 = torch.ones(size=out1.shape, dtype=torch.float, device=device)

# Training

criterion = nn.BCELoss()

optimizerD = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, beta2))

writer = SummaryWriter()

for epoch in range(NUM_EPOCHS+1):
    print(f"Training epoch {epoch+1}")
    for images,_ in iter(dataloader_train):
        # ========= Train Discriminator ===========
        # Train on real data
        # Maximize log(D(x,y)) <- maximize D(x,y)
        model_D.zero_grad()
        
        inputs = images[:,:,:,:256].to(device) # input image data
        targets = images[:,:,:,256:].to(device) # real targets data
        
        real_data = torch.cat([inputs, targets], dim=1).to(device)
        outputs = model_D(real_data) # label "real" data
        labels = torch.ones(size = outputs.shape, dtype=torch.float, device=device)
        
        lossD_real = 0.5 * criterion(outputs, labels) # divide the objective by 2 -> slow down D
        lossD_real.backward()
        
        # Train on fake data
        # Maximize log(1-D(x,G(x))) <- minimize D(x,G(x))
        gens = model_G(inputs).detach()
         
        fake_data = torch.cat([inputs, gens], dim=1) # generated image data
        outputs = model_D(fake_data)
        labels = torch.zeros(size = outputs.shape, dtype=torch.float, device=device) # label "fake" data
        
        lossD_fake = 0.5 * criterion(outputs, labels) # divide the objective by 2 -> slow down D
        lossD_fake.backward()
        
        optimizerD.step()
        
        # ========= Train Generator x2 times ============
        # maximize log(D(x, G(x)))
        for i in range(2):
            model_G.zero_grad()
            
            gens = model_G(inputs)
            
            gen_data = torch.cat([inputs, gens], dim=1) # concatenated generated data
            outputs = model_D(gen_data)
            labels = torch.ones(size = outputs.shape, dtype=torch.float, device=device)
            
            lossG = criterion(outputs, labels) + L1_lambda * torch.abs(gens-targets).sum()
            lossG.backward()
            optimizerG.step()

    # Visualization with TensorBoard

    writer.add_scalar('Loss/Discriminator_loss_on_real_data', lossD_real.item(), epoch)
    writer.add_scalar('Loss/Discriminator_loss_on_fake_data', lossD_fake.item(), epoch)
    writer.add_scalar('Loss/Generator_loss', lossG.item(), epoch)
    writer.add_image('Images/1_Sat', inputs.detach().cpu()[0], epoch)
    writer.add_image('Images/2_Fake_Map', gens.detach().cpu()[0], epoch)
    writer.add_image('Images/3_Real_Map', targets.detach().cpu()[0], epoch)

    # Saving trained

    if(epoch%20==0):
        torch.save(model_G, "./trained_networks/old/generator_epoch_" + str(epoch) + ".pth")
        torch.save(model_D, "./trained_networks/old/discriminator_epoch_" + str(epoch) + ".pth")
    
    torch.save(model_G, "./trained_networks/generator_last.pth")
    torch.save(model_D, "./trained_networks/discriminator_last.pth")


writer.close()
    
print("Done!")

