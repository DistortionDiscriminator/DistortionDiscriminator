import torch
import tools
import torchaudio
eps = 1e-8

def complex_mse_loss(x, y):
    loss = torch.abs(torch.real(x) - torch.real(y) + eps).pow(2) + \
        torch.abs(torch.imag(x) - torch.imag(y)).pow(2)
    loss = torch.mean(loss)
    return loss

def magnitude_mse_loss(x, y):
    loss = torch.abs(x - y + eps).pow(2)
    loss = torch.mean(loss)
    return loss


def spectrum_compressed_loss(est_wav, target_wav):
    est_cmp = tools.stft_22050(est_wav, compress_factor=0.3)
    target_cmp = tools.stft_22050(target_wav, compress_factor=0.3)
    loss = 1 * magnitude_mse_loss(torch.abs(est_cmp), torch.abs(target_cmp)) + \
            1 * complex_mse_loss(est_cmp, target_cmp) 
    return loss

def spectral_convergence_loss(est_wav, target_wav):
    est_abs = torch.abs(tools.stft_22050(est_wav, compress_factor=1))
    target_abs = torch.abs(tools.stft_22050(target_wav, compress_factor=1))
    return torch.norm(target_abs - est_abs, p="fro") / torch.norm(target_abs, p="fro")
    
def regression_loss(est_wav, target_wav):
    loss = spectrum_compressed_loss(est_wav, target_wav)
    return loss

## GAN
def adversarial_loss(score, label):
    # i: multi_resolution
    # j: multi_band
    loss_report = []
    loss_total = 0
    for i in range(len(score)):
        score_list = score[i]
        loss = 0
        for j in range(len(score_list)):
            loss += torch.mean((score_list[j] - label) ** 2)    # B, 1, 64, T/8
        loss_report.append(loss)
        loss_total += loss
    return loss_report, loss_total

def feature_match_loss(est_fmaps, clean_fmaps):
    loss_report = []
    loss_total = 0
    for i in range(len(est_fmaps)):
        est_fmaps_list = est_fmaps[i]
        clean_fmaps_list = clean_fmaps[i]
        loss = 0
        for j in range(len(est_fmaps_list)):
            loss += torch.mean(torch.abs(est_fmaps_list[j] - clean_fmaps_list[j] + 1e-8))
        loss_report.append(loss)
        loss_total += loss
    return loss_report, loss_total

def artifact_adversarial_loss(score, label):
    # i: multi_resolution
    # j: multi_band
    # label: (B, 5, 1, 1)
    loss_report = []
    loss_total = 0
    for i in range(len(score)):
        score_list = score[i]
        loss = 0
        for j in range(len(score_list)):
            label
            loss += torch.mean((score_list[j] - label) ** 2)    # B, 5, 64, T/8
        loss_report.append(loss)
        loss_total += loss
    return loss_report, loss_total