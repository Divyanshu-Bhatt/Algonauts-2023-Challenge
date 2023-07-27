import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ClipRegress(nn.Module):
    def __init__(self, out_dim):
        super(ClipRegress, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class ResNetRegress(nn.Module):
    def __init__(self, out_dim, config="50"):
        super(ResNetRegress, self).__init__()

        if config == "50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif config == "101":
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif config == "152":
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        self.resnet.fc = nn.Linear(2048, out_dim)

    def get_vae_loss(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return self.loss_function(x_recon, x, mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def forward(self, in_features):
        x = self.resnet(in_features)
        return x
