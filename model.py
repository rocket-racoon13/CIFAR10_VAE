import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.channels = [self.args.image_channel] + self.args.conv_channels
        
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
                
        self.decoder = nn.ModuleList()
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
            self.decoder.append(nn.ReLU())
        
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x