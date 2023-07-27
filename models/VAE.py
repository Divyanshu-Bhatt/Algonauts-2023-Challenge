import torch
import torch.nn as nn
from models.extracted_features_model import LeNet_3


class VAE(nn.Module):
    def __init__(self, in_shape, latent_dim):
        super(VAE, self).__init__()

        # self.encoder = LeNet_3(in_shape, out_dim)
        self.encoder_cnn = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.encoder_cnn_mu = nn.Linear(latent_dim, latent_dim)
        self.encoder_cnn_logvar = nn.Linear(latent_dim, latent_dim)

        self.encoder_features = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.encoder_features_mu = nn.Linear(latent_dim, latent_dim)
        self.encoder_features_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_shape),
        )
        self.MSE = nn.MSELoss(reduction="sum")

    def forward(self, in_features):
        # x = self.encoder(in_features)
        x = in_features
        x_cnn = self.encoder_cnn(x)
        x_cnn_mu = self.encoder_cnn_mu(x_cnn)

        x_features = self.encoder_features(x)
        x_features_mu = self.encoder_features_mu(x_features)

        if self.training:
            x_cnn_logvar = self.encoder_cnn_logvar(x_cnn)
            x_features_logvar = self.encoder_features_logvar(x_features)

            x_cnn = self.reparameterize(x_cnn_mu, x_cnn_logvar)
            x_features = self.reparameterize(x_features_mu, x_features_logvar)
        else:
            x_cnn = x_cnn_mu
            x_features = x_features_mu

        x_cnn = self.decoder(x_cnn)
        return x_cnn

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar)

    def get_loss(self, in_features, annealing_factor=1.0):
        # x = self.encoder(in_features)
        x = in_features
        if torch.rand(1) > 0.5:
            x_cnn = self.encoder_cnn(x)
            x_cnn_mu = self.encoder_cnn_mu(x_cnn)
            x_cnn_logvar = self.encoder_cnn_logvar(x_cnn)
            x_mu = x_cnn_mu
            x_logvar = x_cnn_logvar
        else:
            x_features = self.encoder_features(x)
            x_features_mu = self.encoder_features_mu(x_features)
            x_features_logvar = self.encoder_features_logvar(x_features)
            x_mu = x_features_mu
            x_logvar = x_features_logvar

        x = self.reparameterize(x_mu, x_logvar)
        x = self.decoder(x)
        return self.elbo_loss(x_mu, x_logvar, x, in_features, annealing_factor)

    def elbo_loss(self, mu, logvar, x_regenerated, x_original, annealing_factor=1.0):
        MSE_loss = self.MSE(x_regenerated, x_original)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(MSE_loss, KLD)
        return MSE_loss + annealing_factor * KLD
