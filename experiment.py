# Imports

import os
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Set seed for repeatability
torch.manual_seed(42)

# Check if we have GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Z1 - MNIST
print("Starting experiment: Z1 - MNIST")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST("data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten image
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses = []
pbar = tqdm(enumerate(trainloader), total=len(trainloader))
for batch_idx, (data, target) in pbar:
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_description(f"Experiment Z1 - Loss: {loss.item():.4f}")
    
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("MNIST Classification")
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.savefig("experiment_Z1.png", bbox_inches="tight")
print("Saved experiment_Z1.png")

#############################################
# Z2 - XOR of 5 ANDs
print("Starting experiment: Z2 - XOR of 5 ANDs")

concepts = 5
iters = 500
batch_size = 1024

def random_mapping(concepts):
    return torch.randperm(1 << concepts)

def generate_data(concepts, mapping, batch_size=64):
    inputs = torch.randint(0, 2, (batch_size, concepts, 2))
    labels = torch.zeros((batch_size,), dtype=torch.long)
    for i in range(concepts):
        labels = labels ^ ((inputs[:, i, 0] & inputs[:, i, 1]))
    inputs = inputs.reshape(batch_size, -1).float()
    outputs = torch.zeros((batch_size, 2))
    outputs[torch.arange(batch_size), labels] = 1
    return inputs, outputs

class ANDSolver(nn.Module):
    def __init__(self, concepts=2, hidden_dim=64, layers=4):
        super().__init__()
        inputs = 2 * concepts
        outputs = 2
        self.fc_in = nn.Linear(inputs, hidden_dim)
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.fc_out = nn.Linear(hidden_dim, outputs)
        
    def forward(self, x):
        x = self.fc_in(x)
        for fc in self.fc:
            x = x + F.elu(fc(x))
        return self.fc_out(x)

model = ANDSolver(concepts).to(device)
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mapping = random_mapping(concepts)
losses = []
model.train()
pbar = tqdm(range(iters))
for i in pbar:
    inputs, outputs = generate_data(concepts, mapping, batch_size)
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    
    optimizer.zero_grad()
    loss = criterion(F.log_softmax(model(inputs), dim=-1), outputs)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    pbar.set_description(f"Loss: {loss.item():.4f}")
    
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.title("Grokking 5 ANDs")
plt.savefig("experiment_Z2.png", bbox_inches="tight")
print("Saved experiment_Z2.png")

#############################################
# Z6 - Digit GAN
print("Starting experiment: Z6 - Digit GAN")

batch_size = 256
lr = 1e-5
epochs = 200
latent_dim = 100

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainloader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, drop_last=True
)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer = optim.Adam(chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

losses = []
pbar = tqdm(range(epochs))
for epoch in pbar:
    for batch_idx, (real_images, _) in enumerate(trainloader):
        optimizer.zero_grad()
        real_images = real_images.to(device)
        batch_size_curr = real_images.size(0)
        z = torch.randn(batch_size_curr, latent_dim).to(device)

        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)
        
        # Generator loss
        fake_images = generator(z)
        fake_preds = discriminator(fake_images)
        real_preds = discriminator(real_images)
        
        loss = criterion(fake_preds, real_labels)
        loss.backward()
        
        # Discriminator loss
        for param in discriminator.parameters():
                if param.grad is not None:
                    param.grad *= -1
            
        fake_images = generator(z)
        fake_preds = discriminator(fake_images)
        real_preds = discriminator(real_images)
        
        loss = criterion(real_preds, real_labels)
        loss.backward()
        
        optimizer.step()
        losses.append(loss.item())


plt.figure(figsize=(10,6))
plt.plot(losses)
plt.xlabel("Training Iteration")
plt.ylabel("Sum of Losses")
plt.title("Digit GAN")
plt.savefig("experiment_Z6.png", bbox_inches="tight")
print("Saved experiment_Z6.png")
