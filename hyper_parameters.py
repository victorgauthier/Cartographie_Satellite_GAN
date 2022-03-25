import torch

BATCH_SIZE = 1  # Suggested by the original paper of pix2pix
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 200
NGPU = 1
L1_LAMBDA = 0  # <= 0 ---> Loss not calculated
L2_LAMBDA = 100  # <= 0 ---> Loss not calculated
P_LAMBDA = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
