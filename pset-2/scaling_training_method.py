import math
import numpy as np

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

print("="*100)
print("DATA CELL!!!!")
print("="*100)
batch_size = 128

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

trainset = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
testset  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  pin_memory=True)
test_loader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, pin_memory=True)

print("="*100)
print("MODELS AND MINI TRAIN CELL!!!!")
print("="*100)

class MLP(nn.Module):
    def __init__(self, depth, width):
        super(MLP, self).__init__()

        self.initial = nn.Linear(3072, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 10, bias=False)

        self.nonlinearity = lambda x: F.relu(x) * math.sqrt(2)

    def forward(self, x):
        x = x.view(x.shape[0],-1)

        x = self.initial(x)
        x = self.nonlinearity(x)

        for layer in self.layers:
            x = layer(x)
            x = self.nonlinearity(x)

        return self.final(x)

def loop(net, train, eta):
    dataloader  = train_loader   if train else test_loader
    description = "Training... " if train else "Testing... "

    acc_list = []

    for data, target in tqdm.tqdm(dataloader, desc=description):
        data, target = data.cuda(), target.cuda()
        output = net(data)

        loss = output.logsumexp(dim=1).mean() - output[range(target.shape[0]),target].mean() # cross-entropy loss
        acc = (output.max(dim=1)[1] == target).sum() / target.shape[0] # accuracy
        acc_list.append(acc.item())

        if train:
            loss.backward()

            for p in net.parameters():
                fan_out, fan_in = p.shape
                update = torch.sign(p.grad)
                ## <FIXED> START BLOCK that you should modify
                grad_spectral_norm = torch.linalg.matrix_norm(update, ord=2) # what is this was a bias?!?!?? does this just not work :(
                update /= grad_spectral_norm
                update *= math.sqrt(fan_out/fan_in)
                ## <FIXED> END BLOCK that you should modify
                p.data -= eta * update
            net.zero_grad()

    return np.mean(acc_list)

depth = 5
width = 32
from torch.nn.init import orthogonal_
def initialize_matrix(p):
    fan_out, fan_in = p.shape
    ## <FIXED> START BLOCK that you should modify ##
    # is this semi-orthogonal? - from torch.nn.init import orthogonal_
    pp = torch.randn_like(p)
    orthogonal_(pp)
    pp *= math.sqrt(fan_out / fan_in) #torch.randn_like(p) / math.sqrt(fan_in)
    p.data = pp.cuda()
    ## <FIXED> END BLOCK that you should modify

og_train_accs = []
og_test_accs = []
etas = [0.0001, 0.001, 0.01, 0.1, 1.0]
for eta in etas:
    print(f"Training at {width=}, {depth=}, {eta=}")

    net = MLP(depth, width).cuda()

    print("\nNetwork tensor shapes are:\n")
    for name, p in net.named_parameters():
        print(p.shape, '\t', name)
        initialize_matrix(p)

    train_acc = loop(net, train=True,  eta=eta)
    test_acc  = loop(net, train=False, eta=None)
    og_train_accs.append(train_acc)
    og_test_accs.append(test_acc)

    print(f"\nWe achieved train acc={train_acc:.3f} and test acc={test_acc:.3f}\n")
    print("===================================================================\n")

print("="*100)
print("RESULTS CELL!!!!")
print("="*100)

print(f"og_train_accs: {og_train_accs}")
print(f"og_test_accs: {og_test_accs}")
with open("outputs_og.txt", "w") as f:
    f.write(f"og_train_accs: {og_train_accs}\n")
    f.write(f"og_test_accs: {og_test_accs}\n")

print("="*100)

print("="*100)
print("SCALING TRAINING METHOD CELL!!!!")
print("="*100)
depth = 5
width = 4096

# START BLOCK you should set this to the best value of eta from the previous cell
# need to do twice to estimate counterfatual, but they don't ask for that
# etas2 = [0.0001, 0.001, 0.01, 0.1, 1.0]
etas2 = [0.1]
# it's HELLA slow
# assert etas == etas2
# END BLOCK

scaleup_train_accs = []
scaleup_test_accs = []

for eta in etas2:
  print(f"Training at {width=}, {depth=}, {eta=}")

  net = MLP(depth, width).cuda()

  print("\nNetwork tensor shapes are:\n")
  for name, p in net.named_parameters():
      print(p.shape, '\t', name)
      initialize_matrix(p)
    
  net = net.cuda()

  train_acc = loop(net, train=True,  eta=eta)
  test_acc  = loop(net, train=False, eta=None)
  scaleup_train_accs.append(train_acc)
  scaleup_test_accs.append(test_acc)

  print(f"\nWe achieved train acc={train_acc:.3f} and test acc={test_acc:.3f}\n")
  print("===================================================================\n")
print("etas", etas)
print("og training accs", og_train_accs)
print("og test accs", og_test_accs)
print("scaleup training accs", scaleup_train_accs)
print("scaleup test accs", scaleup_test_accs)
with open("outputs_scaleup.txt", "w") as f:
    f.write(f"og_train_accs: {og_train_accs}\n")
    f.write(f"og_test_accs: {og_test_accs}\n")
    f.write(f"scaleup_train_accs: {scaleup_train_accs}\n")
    f.write(f"scaleup_test_accs: {scaleup_test_accs}\n")
