from denoising_diffusion_pytorch import Unet
from inspect import isfunction
from einops import rearrange
from functools import partial
from torch import nn, einsum
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import random
import math

class VAE(nn.Module):
    def __init__(self, z_dims=4, input_size = 784, num_hidden=128):
        super().__init__()
        self.z_dims = z_dims
        self.input_size = input_size

        # FIXED: Create two encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )

        # FIXED: Create the mean and logvar readout layers
        self.mu = nn.Linear(num_hidden, z_dims)
        self.logvar = nn.Linear(num_hidden, z_dims)

        # FIXED: Create the decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # FIXED: Implement the VAE forward function
        # print("done getting input x") # XXX
        encoded = self.encoder(x)
        # print("Done encoding") # XXX
        mu = self.mu(encoded)
        # print("Done mu") # XXX
        logvar = self.logvar(encoded)
        # print("Done logvar") # XXX
        std = torch.exp(0.5 * logvar)
        # print("Done std") # XXX
        eps = torch.randn_like(std) # normal random isotropic
        # print("Done std") # XXX
        z = mu + std * eps
        # print("Done z") # XXX
        reconstructed = self.decoder(z)
        # print("Done reconstructed") # XXX
        return reconstructed, mu, logvar
    
model = VAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
  model.train()
  train_loss = 0
  for (X, _) in tqdm(train_loader):
      X = X.cuda()
      X = X.flatten(start_dim=1) # ignore da batch
      optimizer.zero_grad()
      # print("------") # XXX
      x_prime, mu_z, logvar_z = model(X)
      var_z = torch.exp(logvar_z)

      # FIXED: Calculate loss (ignoring constants that don't affect loss)
      # from: https://www.dropbox.com/scl/fi/jy9u21sqa4zjpkhzyygue/15_gen_models_2.pdf?rlkey=4c7i4evuoxeaskaqu8zzc0qo6&e=1&dl=0
      # print(x_prime.shape, X.shape, mu_z.shape, logvar_z.shape, var_z.shape) # XXX
      reconstruction_losses = 0.5 * (x_prime - X).square() # XXX(Adriano) something is wrong here!
      squash_losses = 0.5 * (mu_z.square() + var_z.square() - logvar_z)
      loss = torch.mean(reconstruction_losses) +torch.mean(squash_losses)

      loss.backward()
      train_loss += loss.item()
      optimizer.step()

  print('Epoch: {} Train Loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))