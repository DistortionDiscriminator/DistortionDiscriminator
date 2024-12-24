import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import numpy as np
import torchaudio
import tools



def WNConv2d(in_channels, out_channels, kernel_size, padding, stride=1, groups=1):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )),
        nn.LeakyReLU(negative_slope=0.2),
    )


class FD_Block(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        channels = channels

        for i in range(1):
            setattr(self, 'convs_{}'.format(i), 
                    nn.ModuleList([
                        WNConv2d(1, channels, (3, 3), stride=(1, 1), padding=(1, 1)),
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1)),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 40
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1),),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 20
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1),),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 10
                    ])
            )
        self.final = nn.utils.weight_norm(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, wav):
        '''
        spec: B, T
        '''
        fmaps = []
        scores = []
        spec = tools.stft_22050(wav, compress_factor=0.3, )                          # B, 512, T
        spec = torch.abs(spec).unsqueeze(1)                                         # B, 1, 512, T

        for layer in self.convs_0:
            spec = layer(spec)      # B, C, 64, T/8
            fmaps.append(spec)      
        score = self.final(spec)    # B, 1, 64, T/8
        scores.append(score)

        return fmaps, scores

class Frequency_Discriminator(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.fd = nn.ModuleList([
            FD_Block(channels=channels),
        ])

    def forward(self, wav):

        fd0_fmaps_list, fd0_scores_list, = [], []

        fmaps, scores, = self.fd[0](wav, )
        fd0_fmaps_list.extend(fmaps)
        fd0_scores_list.extend(scores)

        fd_fmaps = [fd0_fmaps_list]
        fd_score = [fd0_scores_list]
        return fd_fmaps, fd_score,


class AD_Block(nn.Module):
    def __init__(self,channels, out_channels) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.convs = \
                nn.ModuleList([
                    WNConv2d(1, channels, (3, 3), stride=(1, 1), padding=(1, 1)),
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1)),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 40
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1),),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 20
                        WNConv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1),),
                        WNConv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1),),   # 10
                ])
        self.final = nn.utils.weight_norm(
            nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, wav):
        '''
        spec: B, T
        '''
        fmaps = []
        scores = []
        spec = tools.stft_22050(wav, compress_factor=0.3, )    # B, 512, T
        spec = torch.abs(spec).unsqueeze(1)                    # B, 1, 512, T

        for layer in self.convs:
            spec = layer(spec)      # B, C, 64, T/8
            fmaps.append(spec)      
        score = self.final(spec)    # B, 5, 64, T/8
        scores.append(score)
        return fmaps, scores


class Artifact_Discriminator(nn.Module):
    def __init__(self, channels, out_channels) -> None:
        super().__init__()
        self.ad = nn.ModuleList([
            AD_Block(channels=channels, out_channels=out_channels),
        ])

    def forward(self, spec):
        ad0_fmaps_list, ad0_scores_list, = [], []

        fmaps, scores, = self.ad[0](spec, )
        ad0_fmaps_list.extend(fmaps)
        ad0_scores_list.extend(scores)

        ad_fmaps = [ad0_fmaps_list,]
        ad_scores = [ad0_scores_list,]
        return ad_fmaps, ad_scores,

