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

def plot_latents(model, i=0, j=1):
  # FIXED: Plot the image grid
  model_device = next(p.device for p in model.parameters())
  image_np = np.empty((280, 280))
  for x in range(0, 280, 28): # each is a 28 x 28 image
    for y in range(0, 280, 28):
      z = torch.randn(model.z_dims).to(model_device)
      decoded = model.decoder(z)
      assert decoded.shape == (28 * 28, )
      image_np[x:x+28, y:y+28] = decoded.detach().cpu().numpy().reshape(28, 28)
  plt.imshow(image_np)
  plt.show()

def plot_zs(model):
  for i in range(model.z_dims):
    for j in range(model.z_dims):
      if i!=j:
        plot_latents(model, i, j)

plot_zs(model)