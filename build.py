from typing import Any
import numpy as np
import os
import subprocess
import pysepm
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
import torch.optim
import torchaudio
from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import grad_norm
from torch import istft, stft

import data, model, losses, discriminators
from tools import *

CLEAN_LABEL = 1
ESTIM_LABEL = 0

class LightningNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.joint_train = True

        self.generator = model.Net(n_dim=args['n_dim'],)

        if self.args['use_gan'] and self.args['resume'] == 'None':    
            self.generator = load_pretrained_lightning_model(self.generator, args['pretrained'], 'generator')
        
        if self.args['use_fd']:
            self.fd = discriminators.Frequency_Discriminator(channels=64)
            self.d_interval = args['d_interval']

        if self.args['use_ad']:
            if self.args['ad_class'] == 5:
                self.finer = False
            elif self.args['ad_class'] == 9:
                self.finer = True
            else:
                raise ValueError
            self.ad = discriminators.Artifact_Discriminator(channels=64, out_channels=self.args['ad_class'])
            self.d_interval = args['d_interval']
    
    def training_step(self, batch, batch_idx):
        audio, clean, artifact = batch
        if self.args['use_gan']:
            optimizer_g, optimizer_d = self.optimizers()
        else:
            optimizer_g = self.optimizers()

        est_cmp, est_wav = self.generator(audio)
        
        regression_loss = losses.regression_loss(est_wav, clean)
        self.log("reg/train_regression_loss", regression_loss, 
                on_step=False, on_epoch=True, batch_size=self.args['batch_size'], sync_dist=True)

        generator_loss = regression_loss
        
        if self.args['use_gan']:
            if self.args['use_fd']:
                est_fd_fmaps, est_fd_score = self.fd(est_wav)
                cln_fd_fmaps, cln_fd_score = self.fd(clean)
                adv_loss_report, adv_loss_total = losses.adversarial_loss(est_fd_score, CLEAN_LABEL)
                fm_loss_report, fm_loss_total = losses.feature_match_loss(est_fd_fmaps, cln_fd_fmaps)
                generator_loss += self.args['adv_weight'] * adv_loss_total + self.args['fm_weight'] * fm_loss_total
                for i in range(len(fm_loss_report)):
                    self.log('feature_match/fd{}'.format(i), fm_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                    self.log('G_adv/fd{}'.format(i), adv_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
            
            if self.args['use_ad']:
                est_ad_fmaps, est_ad_score = self.ad(est_wav)
                cln_ad_fmaps, cln_ad_score = self.ad(clean)
                clean_score = torch.ones(size=est_ad_score[0][0].shape, device=est_ad_score[0][0].device)
                adv_loss_report, adv_loss_total = losses.artifact_adversarial_loss(est_ad_score, clean_score)
                fm_loss_report, fm_loss_total = losses.feature_match_loss(est_ad_fmaps, cln_ad_fmaps)
                generator_loss += self.args['adv_weight'] * adv_loss_total + self.args['fm_weight'] * fm_loss_total
                for i in range(len(fm_loss_report)):
                    self.log('feature_match/ad{}'.format(i), fm_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                    self.log('G_adv/ad{}'.format(i), adv_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
            
        optimizer_g.zero_grad()
        self.manual_backward(generator_loss)
        total_norm = 0
        for p in self.generator.parameters():
            if p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm/generator", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)

        try:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1, error_if_nonfinite=True)
        except:
            print('G: anamoly, skip this batch')
            return 
        optimizer_g.step()

        if (self.args['use_fd'] and (batch_idx % self.d_interval == 0)) or \
            (self.args['use_ad'] and (batch_idx % self.d_interval == 0)):

            discriminator_loss = 0
            if (self.args['use_fd']) and (batch_idx % self.d_interval == 0):
                est_fd_fmaps, est_fd_score = self.fd(est_wav.detach())
                cln_fd_fmaps, cln_fd_score = self.fd(clean)

                adv_loss_report_0, adv_loss_total_0 = losses.adversarial_loss(est_fd_score, ESTIM_LABEL)
                adv_loss_report_1, adv_loss_total_1 = losses.adversarial_loss(cln_fd_score, CLEAN_LABEL)

                frequency_d_loss = 0.5 * adv_loss_total_0 + 0.5 * adv_loss_total_1
                discriminator_loss += frequency_d_loss

                for i in range(len(adv_loss_report_0)):
                    self.log('D_adv/FD{}'.format(i), 0.5 * adv_loss_report_0[i] + 0.5 * adv_loss_report_1[i], \
                            on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                    
            if (self.args['use_ad']) and (batch_idx % self.d_interval == 0):
                est_ad_fmaps, est_ad_score = self.ad(est_wav.detach())
                cln_ad_fmaps, cln_ad_score = self.ad(clean)

                # estim_score: (B, 5, 1, 1)
                estim_score = artifact_to_binary(artifact, device=est_ad_score[0][0].device, finer=self.finer)
                clean_score = torch.ones(size=est_ad_score[0][0].shape, device=est_ad_score[0][0].device)

                adv_loss_report_0, adv_loss_total_0 = losses.artifact_adversarial_loss(est_ad_score, estim_score)
                adv_loss_report_1, adv_loss_total_1 = losses.artifact_adversarial_loss(cln_ad_score, clean_score)

                artifact_d_loss = 0.5 * adv_loss_total_0 + 0.5 * adv_loss_total_1
                discriminator_loss += artifact_d_loss

                for i in range(len(adv_loss_report_0)):
                    self.log('D_adv/AD{}'.format(i), 0.5 * adv_loss_report_0[i] + 0.5 * adv_loss_report_1[i], \
                            on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
            
            optimizer_d.zero_grad()
            self.manual_backward(discriminator_loss)

            if (self.args['use_fd']):
                total_norm = 0
                for p in self.fd.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.log("grad_norm/FD", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                try:
                    torch.nn.utils.clip_grad_norm_(self.fd.parameters(), max_norm=1, error_if_nonfinite=True)
                except:
                    print('FD: anamoly, skip this batch')
                    return 
            if (self.args['use_ad']):
                total_norm = 0
                for p in self.ad.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.log("grad_norm/AD", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                try:
                    torch.nn.utils.clip_grad_norm_(self.ad.parameters(), max_norm=1, error_if_nonfinite=True)
                except:
                    print('AD: anamoly, skip this batch')
                    return   
            if (self.args['use_md']):
                total_norm = 0
                for p in self.md.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.log("grad_norm/MD", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                try:
                    torch.nn.utils.clip_grad_norm_(self.md.parameters(), max_norm=1, error_if_nonfinite=True)
                except:
                    print('MD: anamoly, skip this batch')
                    return    
            optimizer_d.step()    

    def validation_step(self, batch, batch_idx):
        audio, clean, artifact = batch

        est_cmp, est_wav = self.generator(audio)
        
        regression_loss = losses.regression_loss(est_wav, clean)
        self.log("reg/val_regression_loss", regression_loss, 
                 on_step=False, on_epoch=True, batch_size=self.args['batch_size'], sync_dist=True) 
    

    def configure_optimizers(self):
        params_list = []
        params_list.append({'params': self.generator.parameters()})

        if self.args['use_embed'] and self.joint_train:
            params_list.append({'params': self.extractor.parameters()})

        self.optimizer_g = torch.optim.Adam(params_list,
                                            lr=self.args['g_lr'],
                                            betas=(0.5, 0.9))
        if not self.args['use_gan']:
            return self.optimizer_g
        else:
            params_list = []
            if self.args['use_fd']:
                params_list.append({'params': self.fd.parameters()})
            if self.args['use_ad']:
                params_list.append({'params': self.ad.parameters()})
            if self.args['use_md']:
                params_list.append({'params': self.md.parameters()})
            self.optimizer_d = torch.optim.Adam(params_list,
                                            lr=self.args['d_lr'],
                                            betas=(0.5, 0.9))
            return [self.optimizer_g, self.optimizer_d]
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if (self.global_step + 1)%20e3==0:
            self.optimizer_g.param_groups[0]["lr"] *= 0.98
            if self.args['use_gan']:
                for param_group in self.optimizer_d.param_groups:
                    param_group["lr"] *= 0.98
        self.log('lr/lr_g',self.optimizer_g.param_groups[0]["lr"], sync_dist=True)
        if self.args['use_gan']:
            self.log('lr/lr_d',self.optimizer_d.param_groups[0]["lr"], sync_dist=True)


def build_callbacks(args):
    my_callbacks = [
        callbacks.ModelSummary(max_depth=2),
        callbacks.ModelCheckpoint(
            dirpath=os.path.join('log', args['dir']),
            save_top_k=-1,
            every_n_epochs=1,
            monitor="epoch",
            mode="max",
            filename='{epoch}',
            save_last=True
        ),
    ]
    logger = loggers.TensorBoardLogger(
        save_dir='log',
        name='',
        version=args['dir']
    )
    return my_callbacks, logger

def build_trainer(args):
    my_callbacks, logger = build_callbacks(args)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args['n_gpu'],
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        log_every_n_steps=50,
        callbacks=my_callbacks,
        max_epochs=args['max_epochs'],
        num_sanity_val_steps=0,
        precision=32,
        detect_anomaly=False,
        sync_batchnorm=False,
    )
    return trainer


    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        channel = 48
        self.extractor = model_embed.Extractor(channel=channel)
        self.reconstructor = model_embed.Reconstructor(n_dim=16, feature_dim=channel)
        if self.args['use_fd']:
            self.fd = discriminators.Frequency_Discriminator(channels=64)
            self.d_interval = args['d_interval']
        print('args[\'resume\']==', args['resume'])
    
    def training_step(self, batch, batch_idx):
        if self.args['use_gan']:
            optimizer_g, optimizer_d = self.optimizers()
        else:
            optimizer_g = self.optimizers()

        audio, clean, artifact = batch
        feature = self.extractor(audio)
        est_log_mel, audio_log_mel = self.reconstructor(clean, audio, feature)
        recons_loss = losses.regression_loss(est_log_mel, audio_log_mel)
        self.log("embed/train_reconstrct_loss", recons_loss, 
                on_step=False, on_epoch=True, batch_size=self.args['batch_size'], sync_dist=True)
        
        if self.args['use_gan'] and self.args['use_fd'] and self.current_epoch > 9:
            est_fd_fmaps, est_fd_score = self.fd(est_log_mel)
            cln_fd_fmaps, cln_fd_score = self.fd(audio_log_mel)
            adv_loss_report, adv_loss_total = losses.adversarial_loss(est_fd_score, CLEAN_LABEL)
            fm_loss_report, fm_loss_total = losses.feature_match_loss(est_fd_fmaps, cln_fd_fmaps)
            recons_loss += self.args['adv_weight'] * adv_loss_total + self.args['fm_weight'] * fm_loss_total
            for i in range(len(fm_loss_report)):
                self.log('feature_match/fd{}'.format(i), fm_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                self.log('G_adv/fd{}'.format(i), adv_loss_report[i], on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
            
        optimizer_g.zero_grad()
        self.manual_backward(recons_loss)

        total_norm = 0
        for p in self.extractor.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm/extractor", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
        try:
            torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=1, error_if_nonfinite=True)
        except:
            print('extractor: anamoly, skip this batch')
            return 
        
        total_norm = 0
        for p in self.reconstructor.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm/reconstructor", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
        try:
            torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), max_norm=1, error_if_nonfinite=True)
        except:
            print('reconstructor: anamoly, skip this batch')
            return 
        
        optimizer_g.step()

        if self.args['use_gan'] and self.current_epoch > 9:
            discriminator_loss = 0
            if (self.args['use_fd']) and (batch_idx % self.d_interval == 0):
                est_fd_fmaps, est_fd_score = self.fd(est_log_mel.detach())
                cln_fd_fmaps, cln_fd_score = self.fd(audio_log_mel)

                adv_loss_report_0, adv_loss_total_0 = losses.adversarial_loss(est_fd_score, ESTIM_LABEL)
                adv_loss_report_1, adv_loss_total_1 = losses.adversarial_loss(cln_fd_score, CLEAN_LABEL)

                frequency_d_loss = 0.5 * adv_loss_total_0 + 0.5 * adv_loss_total_1
                discriminator_loss += frequency_d_loss

                for i in range(len(adv_loss_report_0)):
                    self.log('D_adv/FD{}'.format(i), 0.5 * adv_loss_report_0[i] + 0.5 * adv_loss_report_1[i], \
                            on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                    
            optimizer_d.zero_grad()
            self.manual_backward(discriminator_loss)

            if (self.args['use_fd']):
                total_norm = 0
                for p in self.fd.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.log("grad_norm/FD", total_norm, on_step=True, on_epoch=False, batch_size=self.args['batch_size'], sync_dist=False)
                try:
                    torch.nn.utils.clip_grad_norm_(self.fd.parameters(), max_norm=1, error_if_nonfinite=True)
                except:
                    print('FD: anamoly, skip this batch')
                    return 
    
    def validation_step(self, batch, batch_idx):
        audio, clean, artifact = batch
        feature = self.extractor(audio)
        est_log_mel, audio_log_mel = self.reconstructor(clean, audio, feature)
        loss = losses.regression_loss(est_log_mel, audio_log_mel)
        self.log("embed/val_reconstrct_loss", loss, 
                on_step=False, on_epoch=True, batch_size=self.args['batch_size'], sync_dist=True)
    
    def configure_optimizers(self):
        self.optimizer_g = torch.optim.Adam([
                                            {'params': self.extractor.parameters()}, 
                                            {'params': self.reconstructor.parameters()}
                                        ],
                                        lr=self.args['g_lr'],
                                        betas=(0.5, 0.9))
        if not self.args['use_gan']:
            return self.optimizer_g
        else:
            params_list = []
            if self.args['use_fd']:
                params_list.append({'params': self.fd.parameters()})
            self.optimizer_d = torch.optim.Adam(params_list,
                                            lr=self.args['d_lr'],
                                            betas=(0.5, 0.9))
            return [self.optimizer_g, self.optimizer_d]