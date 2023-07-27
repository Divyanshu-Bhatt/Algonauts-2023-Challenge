import torch
import torch.nn as nn
from models.VAE_encoded_loss import VariationalAutoEncoder

class CrossVAE(nn.Module):
    """
    CrossVAE model architecture which takes ResNet Layer 3 Features and clip features as input
    Encoded Loss is used for training the model
    """
    def __init__(self, in_shape, out_dim, encoded_loss=False, fmri_encoder_path=None):
        super(CrossVAE, self).__init__()

        # Processing on Layer3 Extracted ResNet Features 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Concatenating the above features with the clip features
        self.linear_encoder = nn.Linear(4096 + 768, out_dim)

        # VAE Encoder 
        self.encoder_cnn = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
        )
        self.encoder_cnn_mu = nn.Linear(512, 512)
        self.encoder_cnn_logvar = nn.Linear(512, 512)

        # VAE Encoder (for decoder alignment)
        self.encoder_features = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
        )
        self.encoder_features_mu = nn.Linear(512, 512)
        self.encoder_features_logvar = nn.Linear(512, 512)

        # VAE Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, out_dim),
        )
        self.MSE = nn.MSELoss(reduction="sum")
        
        self.encoded_loss = encoded_loss
        if self.encoded_loss:
            if fmri_encoder_path is None:
                print("fmri_encoder_path is None")
                raise ValueError
            self.fmri_encoder = VariationalAutoEncoder(out_dim)
            self.fmri_encoder.load_state_dict(torch.load(fmri_encoder_path))

            for param in self.fmri_encoder.parameters():
                param.requires_grad = False

    def forward(self, in_features, clip_features):
        x = self.encoder(in_features)
        x = x.view(-1, 4096)
        x = torch.cat((x, clip_features), dim=1)
        x = self.linear_encoder(x)

        x_cnn = self.encoder_cnn(x)
        x_cnn_mu = self.encoder_cnn_mu(x_cnn)
        x_cnn_logvar = self.encoder_cnn_logvar(x_cnn)

        if self.training:
            x_cnn = self.reparameterize(x_cnn_mu, x_cnn_logvar)
        else:
            x_cnn = x_cnn_mu

        x_cnn = self.decoder(x_cnn)
        return x_cnn

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar)

    def get_loss(self, in_features, clip_features, target, annealing_factor=1.0):
        # encoder_features helps in decoder alignment
        if torch.rand(1) > 0.5:
            x = self.encoder(in_features)
            x = x.view(-1, 4096)
            x = torch.cat((x, clip_features), dim=1)
            x = self.linear_encoder(x)
            x_cnn = self.encoder_cnn(x)
            x_cnn_mu = self.encoder_cnn_mu(x_cnn)
            x_cnn_logvar = self.encoder_cnn_logvar(x_cnn)
            x_mu = x_cnn_mu
            x_logvar = x_cnn_logvar
        else:
            x_features = self.encoder_features(target)
            x_features_mu = self.encoder_features_mu(x_features)
            x_features_logvar = self.encoder_features_logvar(x_features)
            x_mu = x_features_mu
            x_logvar = x_features_logvar

        x = self.reparameterize(x_mu, x_logvar)
        x = self.decoder(x)

        loss = self.elbo_loss(x_mu, x_logvar, x, target, annealing_factor)
        if self.encoded_loss:
            loss += self.get_encoded_loss(x, target)
        return loss

    def elbo_loss(self, mu, logvar, x_regenerated, x_original, annealing_factor=1.0):
        MSE_loss = self.MSE(x_regenerated, x_original)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE_loss + annealing_factor * KLD

    def get_encoded_loss(self, x_regenerated, target):
        z_regenerated = self.fmri_encoder(x_regenerated)
        z_target = self.fmri_encoder(target)
        return self.MSE(z_regenerated, z_target)