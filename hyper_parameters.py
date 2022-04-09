import torch

BATCH_SIZE = 1  # Suggested by the original paper of pix2pix
LEARNING_RATE = 0.0002 # Suggested by the original paper of pix2pix
BETA1 = 0.5 # Suggested by the original paper of pix2pix
BETA2 = 0.999 # Suggested by the original paper of pix2pix
NUM_EPOCHS = 200 
NGPU = 1
L1_LAMBDA = 0  # <= 0 ---> Loss not calculated
L2_LAMBDA = 0  # <= 0 ---> Loss not calculated
P_LAMBDA = 0 # <= 0 ---> Loss not calculated

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # If cuda is installed
