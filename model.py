import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from datetime import datetime


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.image_width = self.args.image_width
        self.image_height = self.args.image_height
        self.channels = [self.args.image_channel] + self.args.conv_channels
        self.latent_dim = self.args.latent_dim
        
        # encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.args.conv_channels)):
            self.encoder.append(
                nn.Conv2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=self.args.kernel_size[i],
                    stride=self.args.enc_stride[i],
                    padding=self.args.enc_padding[i]
                )
            )
            self.encoder.append(nn.ReLU())
        
        # mu and logvar
        self.feature_dim = self.channels[-1]*self.image_width*self.image_height
        self.mu = nn.Linear(self.feature_dim, self.latent_dim)
        self.logvar = nn.Linear(self.feature_dim, self.latent_dim)
        
        # decoder
        self.decoder = nn.ModuleList()
        self.decFC = nn.Linear(self.latent_dim, self.feature_dim)
        for i in range(1, len(self.args.conv_channels)+1):
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels=self.channels[-i],
                    out_channels=self.channels[-(i+1)],
                    kernel_size=self.args.kernel_size[-i],
                    stride=self.args.dec_stride[(i-1)],
                    padding=self.args.dec_padding[(i-1)]
                )
            )
            if i < len(self.args.conv_channels):
                self.decoder.append(nn.ReLU())
    
    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = x.view(-1, self.feature_dim)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # eps ~ N(0, 1)
        return mu + eps * std
    
    def decode(self, z):
        x = F.relu(self.decFC(z))
        x = x.view(-1, self.channels[-1], self.image_height, self.image_width)
        for layer in self.decoder:
            x = layer(x)
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar