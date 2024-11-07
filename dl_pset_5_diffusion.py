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

NUM_EPOCHS = 5
TOTAL_TIMESTEPS = 200
scaled_size = 32

# Create the Unet model
model = Unet(scaled_size, channels=1, dim_mults=(1, 2, 4,)).to(device)

# We will use the following variance schedule for our diffusion model
beta_t = torch.linspace(0.0001, 0.02, TOTAL_TIMESTEPS)

def forward_sample(x_0: torch.Tensor, t: torch.Tensor | int, e: torch.Tensor):
    # FIXME: calculate x_t from x_0, e, and t using the equation provided in the homework
    assert len(t.shape) == 1
    x_t = x_0
    assert len(x_0.shape) >= 2, x_0.shape
    assert x_0.shape[0] == t.shape[0]
    for batch_idx in range(len(t)):
      for T in range(0, t[batch_idx]):
        x_t[batch_idx] = x_t[batch_idx] + torch.sqrt(1 - beta_t[T]) * e[batch_idx] # XXX why are we using the same e every time?
    return x_t

optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(NUM_EPOCHS):
    train_losses = []
    for (X, y) in tqdm(train_loader):
      optimizer.zero_grad()
      batch_size = X.shape[0]
      batch = X.to(device)

      # Sample a batch of times for training
      t = torch.randint(0, TOTAL_TIMESTEPS, (batch_size,), device=device).long()

      # Calculate the loss
      e = torch.randn_like(batch)
      x_t = forward_sample(batch, t, e)
      e_pred = model(x_t, t)
      loss = F.mse_loss(e, e_pred)

      # Gradient step
      train_losses.append(loss.item())
      loss.backward()
      optimizer.step()

    print("Epoch: {} Loss: {}".format(epoch, np.mean(train_losses)))

    def reverse_sample(model, x, t, t_index):

    # FIXED: Using beta_t, calculate beta_tilde_t using equation 5 from the pset document
    a_t = 1 - beta_t[t_index]
    b_t = beta_t[t_index]

    a_topbar = 1
    for s in range(0, t_index):
      a_topbar *= 1 - beta_t[s] # a_s = 1 - beta_s
    a_topbar_t_m1 = a_topbar
    a_topbar_t = a_topbar * (1 - b_t)

    scale = (1 - a_topbar_t_m1) / (1 - a_topbar_t)
    ########################

    beta_tilde_t = beta_t[t_index] * scale
    ########################

    # FIXED: Using beta_t, calculate mu_tilde_t using equation 6 from the pset document
    mu_tilde_t = (torch.sqrt(a_t) * scale) * x + (b_t * scale) * e
    ########################


    e = torch.randn_like(x)
    return mu_tilde_t + torch.sqrt(beta_tilde_t) * e


@torch.no_grad()
def sample(model):
    shape = (1, 1, scaled_size, scaled_size)
    img = torch.randn(shape, device=device)
    imgs = [img.cpu().numpy()]
    for i in reversed(range(0, TOTAL_TIMESTEPS)):
        img = reverse_sample(model, img, i)
        imgs.append(img.cpu().numpy())
    return imgs

samples = sample(model)
# FIXME: Plot the model inference over time using the samples calculated above