from typing import *
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
import contextlib

import torch
from torch import nn, optim
from torch.nn import functional as F

device = torch.device('cuda:0')
assert torch.cuda.is_available()

import torchvision
from IPython import display
from PIL import Image

from tqdm.auto import tqdm

# load data

data = np.load('/mnt/align3_drive/adrianoh/dl-pset4/content/shapes_dataset_6s898.npy', allow_pickle=True).item()
train_indices, val_indices = torch.arange(data['imgs'].shape[0]).split([64000 - 24000, 24000])


class Dataset:
    r'''
    Our dataset object for loading shapes dataset.
    '''

    def __init__(self, split: str, transform=None, num_samples: int = 1):
        r'''
        split (str): Whether to load training of validation images. Should be 'train' or 'val'.
        transform: Transformations on raw data, e.g., augmentations and/or normalization.
                   `to_tensor` and normalization is called automatically.
                   No need to explicitly pass in `ToTensor()` or `Normalize()`.
        num_samples (int):Number of transformed versions to return for each sample.
                           For autoencoder, this is 1. For contrastive, this is 2.
        '''
        self.split = split
        if split == 'train':
            self.indices = train_indices
        else:
            assert split == 'val'
            self.indices = val_indices
        self.num_samples = num_samples
        if transform is None:
            transform = lambda x: x
        self.transform = transform

    def get_augs(self, idx, num_samples):
        img = torchvision.transforms.functional.to_tensor(data['imgs'][self.indices[idx]])
        return tuple(self.transform(img).clamp(0, 1) for _ in range(num_samples))

    def __getitem__(self, idx):
        r'''
        Fetech the data at index `idx`
        '''
        return tuple(tensor.sub(0.5).div(0.5) for tensor in self.get_augs(idx, num_samples=self.num_samples))

    def visualize(self, idx, num_samples=None):
        r'''
        Visualize the image at index `idx` for `num_samples` times (default to `self.num_samples`).

        These samples will be different if `self.transform` is random.
        '''
        if num_samples is None:
            num_samples = self.num_samples
        f, axs = plt.subplots(1, num_samples, figsize=(1.2 * num_samples, 1.4))
        if num_samples == 1:
            axs = [axs]
        else:
            axs = axs.reshape(-1)
        for ax, tensor in zip(axs, self.get_augs(idx, num_samples)):
            ax.axis('off')
            ax.imshow(tensor.permute(1, 2, 0))
        title = f'{self.split} dataset[{idx}]'
        if num_samples > 1:
            title += f'  ({num_samples} samples)'
        f.suptitle(title, fontsize=17, y=0.98)
        f.tight_layout(rect=[0, 0.03, 1, 0.9])
        return f

    def __len__(self):
        return self.indices.shape[0]

# encoder architecture
class Encoder(nn.Module):
    def __init__(self, latent_dim, normalize: bool = False):
        r'''
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        '''
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 64x64 --> 32x32
        self.conv2 = nn.Conv2d(32, 32*2, 4, stride=2, padding=1) # 32x32 --> 16x16
        self.conv3 = nn.Conv2d(32*2, 32*4, 4, stride=2, padding=1) # 16x16 --> 8x8
        self.conv4 = nn.Conv2d(32*4, 32*8, 4, stride=2, padding=1) # 8x8 --> 4x4
        self.conv5 = nn.Conv2d(32*8, 32*16, 4, stride=2, padding=1) # 4x4 --> 2x2
        self.conv6 = nn.Conv2d(32*16, latent_dim, 4, stride=2, padding=1) # 2x2 --> 1x1
        self.fc = nn.Linear(latent_dim, latent_dim)

        self.nonlinearity = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        x = self.nonlinearity(self.conv6(x).flatten(1))
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x)
        return x

    def extra_repr(self):
        return f'normalize={self.normalize}'


import einops
def train_contrastive(transforms: List, latent_dim: int, *, tau: float = 0.07):
    r'''
    Train encoder with `latent_dim` latent dimensions according
    to the **contrastive** objective described above using temperature
    `tau`.

    Implementation should follow notes above (including negative sampling
    from batch).

    The postive pairs are generated using random augmentations
    specified in `transform`.

    Returns the trained encoder.
    '''

    enc = Encoder(latent_dim, normalize=True).to(device)

    optim = torch.optim.Adam(enc.parameters(), lr=2e-4)

    dataset = Dataset('train', torchvision.transforms.Compose(transforms), num_samples=2)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    num_epochs = 15

    for epoch in range(num_epochs):
        for batch1, batch2 in tqdm(dataloader, desc=f'Epoch {epoch} / {num_epochs}'):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            # batch1: a batched image tensor of shape [B x 3 x 64 x 64]
            # batch2: a batched image tensor of shape [B x 3 x 64 x 64]

            # For each i, p
            #   Positive pairs are (batch1[i], batch2[i])
            #   Negative pairs are (batch1[i], batch2[j]), j != i.
            x1 = enc(batch1)
            x2 = enc(batch2)
            assert x1.shape == x2.shape
            assert len(x1.shape) == 2
            
            # matmul does the work of everything we need, though we have done some of these twice
            loss_table = einops.einsum(x1, x2, 'batch1 dim, batch2 dim -> batch1 batch2')
            loss_table = torch.exp(loss_table / tau) # get all the exponentials
            loss_table = loss_table / loss_table.sum(dim=-1) # normalize the rows - softmax
            
            # FIXME
            loss = -torch.mean(torch.log(loss_table.diagonal())) # each of b gets b-1 negative and 1 positive => 1 term over b items

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'[Contrastive] epoch {epoch: 4d}   loss = {loss.item():.4g}')

    return enc

aug_A = [
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.ColorJitter(hue=.5, brightness=0.3, contrast=0.3, saturation=0.9),
]
interpolation = torchvision.transforms.InterpolationMode.NEAREST
aug_B = [
    torchvision.transforms.Pad(24),
    torchvision.transforms.RandomRotation(degrees=(0, 360)),
    torchvision.transforms.RandomResizedCrop([64, 64], scale=(0.2, 0.6), ratio=(1, 1), interpolation=interpolation),
]
interpolation = torchvision.transforms.InterpolationMode.NEAREST
aug_C = [
    torchvision.transforms.Pad(16),
    torchvision.transforms.RandomRotation(degrees=(0, 360)),
    torchvision.transforms.RandomApply(
        [
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), shear=(-80, 80, -80, 80), interpolation=interpolation),
            torchvision.transforms.RandomPerspective(distortion_scale=0.6,  p=1, interpolation=interpolation),
        ],
        p=0.7,
    ),
    torchvision.transforms.RandomResizedCrop([64, 64], scale=(0.4, 0.8), ratio=(0.2, 5), interpolation=interpolation),
    torchvision.transforms.RandomRotation(degrees=(0, 360)),
]

from pathlib import Path
if __name__ == '__main__':
    save_dir = Path('/mnt/align3_drive/adrianoh/dl-pset4/models')
    save_dir.mkdir(parents=False, exist_ok=False)
    contrastive_enc_aug_A = train_contrastive(transforms=aug_A, latent_dim=128)
    torch.save(contrastive_enc_aug_A.cpu().state_dict(), save_dir / 'contrastive_enc_aug_A.pth')
    contrastive_enc_aug_B = train_contrastive(transforms=aug_B, latent_dim=128)
    torch.save(contrastive_enc_aug_B.cpu().state_dict(), save_dir / 'contrastive_enc_aug_B.pth')
    contrastive_enc_aug_C = train_contrastive(transforms=aug_C, latent_dim=128)
    torch.save(contrastive_enc_aug_C.cpu().state_dict(), save_dir / 'contrastive_enc_aug_C.pth')
