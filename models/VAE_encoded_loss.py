import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_shape):
        super(VariationalAutoEncoder, self).__init__()
        hidden_dim = 1024
        latent_dim = 512

        self.encoder = nn.Sequential(nn.Linear(in_shape, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_shape),
        )

        self.MSE = nn.MSELoss(reduction="sum")

    def forward(self, in_features):
        x = self.encoder(in_features)
        x_mu = self.mu(x)
        x_logvar = self.logvar(x)

        if self.training:
            x = self.reparameterize(x_mu, x_logvar)
        else:
            x = x_mu

        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar)

    def get_loss(self, in_features, annealing_factor=1.0):
        x = self.encoder(in_features)
        x_mu = self.mu(x)
        x_logvar = self.logvar(x)

        x = self.reparameterize(x_mu, x_logvar)
        x = self.decoder(x)
        return self.elbo_loss(x_mu, x_logvar, x, in_features, annealing_factor)

    def elbo_loss(self, mu, logvar, x_regenerated, x_original, annealing_factor=1.0):
        MSE_loss = self.MSE(x_regenerated, x_original)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE_loss, annealing_factor * KLD

    def encode(self, in_features):
        x = self.encoder(in_features)
        x_mu = self.mu(x)
        return x_mu
