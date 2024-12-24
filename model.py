
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stft, istft
import torchaudio
import tools

eps = 1e-8 

########################
class TCN(nn.Module):
    def __init__(self,
                cin,
                K=3,
                dila=1
                ):
        super().__init__()
        channel = cin // 4 
        self.pconv1 = nn.Sequential(
            nn.Conv1d(cin, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.PReLU(channel),
        )
        dila_pad = dila * (K - 1)
        self.dila_conv = nn.Sequential(
            nn.ConstantPad1d((dila_pad, 0), 0.0),
            nn.Conv1d(channel, channel, K, 1, dilation=dila),
            nn.BatchNorm1d(channel),
            nn.PReLU(channel)
        )
        self.dila_gate_conv = nn.Sequential(
            nn.ConstantPad1d((dila_pad, 0), 0.0),
            nn.Conv1d(channel, channel, K, 1, dilation=dila),
            nn.BatchNorm1d(channel),
            nn.PReLU(channel)
        )
        self.pconv2 = nn.Conv1d(channel, cin, kernel_size=1)

    def forward(self, inps):
        """
            inp: B x (C x F) x T
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs) * torch.sigmoid(self.dila_gate_conv(outs))
        outs = self.pconv2(outs)
        return outs + inps


class StackedTCN(nn.Module):
    def __init__(self,
                 cin,
                 K=3,
                 depth=6,
                 ):
        super().__init__()
        self.tcm = nn.ModuleList()
        for i in range(depth):
            self.tcm.append(
                TCN(cin, K, 2**i)
            )

    def forward(self, inp):
        """
            inp: B x (C x F) x T
        """
        out = inp
        for i in range(len(self.tcm)):
            out = self.tcm[i](out)
        return out 


def get_padding_1d(kernel_size, dilation=1):
    # so padding is (T_left, T_right)
    return (
            int((kernel_size*dilation - dilation)),
            0,
        )

def get_padding_2d(kernel_size, dilation=(1, 1)):
    # expect feature map: T, F
    # so padding is (F_top, F_down, T_left, T_right)
    return (
            int((kernel_size[1]*dilation[1] - dilation[1])/2),
            int((kernel_size[1]*dilation[1] - dilation[1])/2),
            int((kernel_size[0]*dilation[0] - dilation[0])),
            0,
        )

class DenseBlock(nn.Module):
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 1
            dense_conv = nn.Sequential(
                nn.ZeroPad2d(get_padding_2d(kernel_size, (dil, 1))),
                nn.Conv2d(self.dense_channel*(i+1), self.dense_channel, kernel_size, dilation=(dil, 1), padding=0),
                nn.InstanceNorm2d(self.dense_channel, affine=True),
                nn.PReLU(self.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x
    
class DenseEncoder(nn.Module):
    def __init__(self, dense_channel, in_channel, depth):
        super(DenseEncoder, self).__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.conv_1 = nn.Sequential(
            nn.ZeroPad2d(get_padding_2d(kernel_size=(3, 3))),
            nn.Conv2d(in_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel),
            )

        self.conv_2 = nn.Sequential(
            DenseBlock(dense_channel=self.dense_channel, depth=self.depth),
            nn.ZeroPad2d(get_padding_2d(kernel_size=(3, 3))),
            nn.Conv2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel)
            )
        
        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d(get_padding_2d(kernel_size=(3, 3))),
            nn.Conv2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel)
            )
        
        self.conv_4 = nn.Sequential(
            nn.ZeroPad2d(get_padding_2d(kernel_size=(3, 3))),
            nn.Conv2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel)
            )

    def forward(self, x):
        '''
        input: (B, C, T, F)
        '''
        x = self.conv_1(x)  # [b, C, T, F]
        x = self.conv_2(x)  # [b, C, T, F/2]
        x = self.conv_3(x)  # [b, C, T, F/4]
        x = self.conv_4(x)  # [b, C, T, F/8]
        return x
    
class DenseDecoder(nn.Module):
    def __init__(self, dense_channel, depth, out_channel):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.out_channel = out_channel

        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(self.dense_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), 
                               padding=(1, 1), output_padding=(0, 0)),
        )

        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2), 
                               padding=(1, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel),
            DenseBlock(dense_channel=self.dense_channel, depth=self.depth),
        )
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2), 
                               padding=(1, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel))
        
        self.conv_4 = nn.Sequential(
            nn.ConvTranspose2d(self.dense_channel, self.dense_channel, kernel_size=(3, 3), stride=(1, 2), 
                               padding=(1, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(self.dense_channel, affine=True),
            nn.PReLU(self.dense_channel))


    def forward(self, x):
        x = self.conv_4(x)  # [B, C, T, F/8]
        x = self.conv_3(x)  # [B, C, T, F/4]
        x = self.conv_2(x)  # [B, C, T, F/2]
        x = self.conv_1(x)  # [B, C, T, F]
        return x
 
class Net(nn.Module):
    def __init__(self, 
                n_dim=64,):
        super().__init__()
        self.ndim = n_dim
        self.encoder = nn.Sequential(
            DenseEncoder(dense_channel=n_dim, in_channel=2, depth=4),
            DenseEncoder(dense_channel=n_dim, in_channel=n_dim, depth=4),
        )
        self.temporal = nn.Sequential(
            StackedTCN(cin=n_dim*8, depth=4),
            StackedTCN(cin=n_dim*8, depth=4),
            StackedTCN(cin=n_dim*8, depth=4),
        )
        self.mask_decoder = nn.Sequential(
            DenseDecoder(dense_channel=n_dim, out_channel=n_dim, depth=4,),
            DenseDecoder(dense_channel=n_dim, out_channel=2, depth=4,),
        )
        self.resi_decoder = nn.Sequential(
            DenseDecoder(dense_channel=n_dim, out_channel=n_dim, depth=4,),
            DenseDecoder(dense_channel=n_dim, out_channel=2, depth=4,),
        )
    def forward(self, audio):
        '''
        input: (B, T)
        output: (B, T)
        '''
        x = tools.stft_22050(audio, compress_factor=0.3, )[:, 1:, :]                          # B, 512, T
        x = torch.stack([torch.real(x), torch.imag(x)], dim=3).permute(0, 3, 2, 1)    # B, 2, T, 512

        b, _ ,t, f = x.shape
        z = self.encoder(x)                                                 # B, C, T, 8
        z = z.permute(0, 1, 3, 2).reshape(b, self.ndim*8, t)                # B, 8*C, T
        z = self.temporal(z)                                                # B, 8*C, T
        z = z.reshape(b, self.ndim, 8, t).permute(0, 1, 3, 2)               # B, C, T, 8

        mask = self.mask_decoder(z)              # B, 2, T, 512  
        resi = self.resi_decoder(z)              # B, 2, T, 512
  
        mask_real, mask_imag = mask[:, 0, :, :], mask[:, 1, :, :]
        mask_mag = torch.sqrt(mask_real ** 2 + mask_imag ** 2 + eps)
        mask_r_phase = mask_real / (mask_mag + eps)
        mask_i_phase = mask_imag / (mask_mag + eps)
        mask_phase = torch.atan2(mask_i_phase, mask_r_phase)

        x_real, x_imag = x[:, 0, :, :], x[:, 1, :, :]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + eps)
        x_r_phase = x_real / (x_mag + eps)
        x_i_phase = x_imag / (x_mag + eps)
        x_phase = torch.atan2(x_i_phase, x_r_phase)

        resi_real, resi_imag = resi[:, 0, :, :], resi[:, 1, :, :]
        s_mag = x_mag * mask_mag
        s_phase = x_phase + mask_phase
        s_real = s_mag * torch.cos(s_phase) + resi_real
        s_imag = s_mag * torch.sin(s_phase) + resi_imag

        s = s_real + 1j * s_imag    # B, T, 512
        s = s.transpose(-1, -2)     # B, 512, T
        s = torch.cat([torch.zeros(size=(b, 1, t), device=s.device), s], dim=1)     # B, 513, T
        est_wav = tools.istft_22050(s, compress_factor=0.3)
        return s, est_wav
    