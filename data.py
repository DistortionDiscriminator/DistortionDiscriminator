import os
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio.io import AudioEffector, CodecConfig
import tools

eps = 1e-8


class AudioDataset(Dataset):
    def __init__(self, hours, args, p=None) -> None:
        super().__init__()
        self.cleanlist = torch.load('data_lists/vctk_wav48_silence_trimmed.pt')
        self.noiselist = torch.load('data_lists/noise.pt')
        self.rirlist = torch.load('data_lists/rir.pt')
        self.hours = hours

        self.length = (300-1) * 480
        
        if p==None:
            self.p = 0.5

        self.prob = {
            'reverb': self.p,
            'noise': self.p,
            'clipping': self.p,
            'color': self.p / (1 - self.p ** 5),
            'resample': self.p,
        }

    def __len__(self):
        return int(self.hours * 3600 / 3)

    def _decide_artifact(self):
        artifact = {}
        for name in self.prob.keys():
            artifact[name] = np.random.choice([False, True], p=[1 - self.prob[name], self.prob[name]])
        artifact['lowpass'] = False
        artifact['lowshelf'] = False
        artifact['highpass'] = False
        artifact['highshelf'] = False
        artifact['peakshelf'] = False
        return artifact
    
    def _load_random_clean(self, cleanlist):
        clean_name = cleanlist[np.random.randint(len(cleanlist))]
        clean = torchaudio.load(clean_name)[0][0]
        while len(clean)<=self.length or torch.max(clean)==0:
            clean = torch.cat([clean,clean],dim=0)
        begin = np.random.randint(0, len(clean) - self.length)
        clean = clean[begin: begin + self.length]
        return clean
    
    def _load_random_noise(self, noiselist):
        noise = torchaudio.load(noiselist[np.random.randint(len(noiselist))])[0][0]
        while len(noise)<=self.length:
            noise = torch.cat([noise,noise],dim=0)
        begin = np.random.randint(0, len(noise) - self.length)
        noise = noise[begin: begin + self.length]
        return noise
    
    def _load_random_rir(self, rirlist):
        while True:
            index = np.random.randint(len(rirlist))
            rir = torchaudio.load(rirlist[index])[0][0]
            rir = rir / torch.norm(rir, p=2)
            if torch.any(rir[:440]>0.5):
                break
        return rir
    
    def _add_noise(self, clean, noise, snr):
        clean_energy = torch.norm(clean, p=2) ** 2
        noise_energy = torch.norm(noise, p=2) ** 2
        A = torch.sqrt(clean_energy / (eps + noise_energy) / (10**(snr / 10)))
        mix = clean + A * noise
        return mix
    
    def _add_rir(self, clean, rir, early=False):
        if early:
            # Early 10ms
            rir[480:] = 0
        mix = tools.convolve(clean, rir)[:len(clean)]
        return mix
    
    def _apply_codec(self, codec_effector, audio):
        return codec_effector.apply(audio.unsqueeze(-1), sample_rate=48000).squeeze(-1)

    def __getitem__(self, index):
        artifact = self._decide_artifact()
        
        clean = self._load_random_clean(self.cleanlist)
        noise = self._load_random_noise(self.noiselist)

        audio = clean

        if artifact['reverb']:
            rir = self._load_random_rir(self.rirlist)
            audio = self._add_rir(audio, rir, early=False)
            clean = self._add_rir(clean, rir, early=True)

        if artifact['noise']==True:
            snr = np.random.randint(-5, 20+1)
            audio = self._add_noise(audio, noise, snr)

        if artifact['color']:
            really_colored = False
            ###### lowpass ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['lowpass'] = True
                freq = np.random.randint(50, 6000)
                audio = torchaudio.functional.lowpass_biquad(audio, sample_rate=48000, cutoff_freq=freq)

            ###### lowshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['lowshelf'] = True
                freq = np.random.randint(50, 700)
                gain = np.random.randint(-12, 12)
                audio = torchaudio.functional.bass_biquad(audio, sample_rate=48000, gain=gain, central_freq=freq)

            ###### highpass ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['highpass'] = True
                freq = np.random.randint(10, 400)
                audio = torchaudio.functional.highpass_biquad(audio, sample_rate=48000, cutoff_freq=freq)

            ###### highshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['highshelf'] = True
                freq = np.random.randint(4500, 7500)
                gain = np.random.randint(-12, 12)
                audio = torchaudio.functional.treble_biquad(audio, sample_rate=48000, gain=gain, central_freq=freq)

            ###### peakshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['peakshelf'] = True
                num_peak = np.random.randint(1, 4+1)
                for _ in range(num_peak):
                    freq = np.random.randint(600, 7400)
                    gain = np.random.randint(-6, 6)
                    audio = torchaudio.functional.equalizer_biquad(audio, sample_rate=48000, center_freq=freq, gain=gain)
            
            if not really_colored:
                artifact['color'] = False
        
        if artifact['resample']: 
            ds = np.random.randint(2, 20+1) * 1000
            lowpass_order = 20
            audio = torchaudio.functional.resample(audio, 48000, ds, lowpass_filter_width=lowpass_order)
            audio = torchaudio.functional.resample(audio, ds, 48000, lowpass_filter_width=lowpass_order)
                
        if artifact['clipping']==True:
            threshold = np.random.uniform(0.1, 0.9)
            audio = torch.clamp(audio, -threshold, threshold)

        audio = torch.clamp(audio, -1, 1)

        audio = torchaudio.functional.resample(audio, 48000, 22050, lowpass_filter_width=6)
        clean = torchaudio.functional.resample(clean, 48000, 22050, lowpass_filter_width=6)

        return audio, clean, artifact
    
def build_dataloaders(args):
    train_loader = DataLoader(AudioDataset(hours=100, args=args), batch_size=args['batch_size'],
                               num_workers=16, shuffle=False, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(AudioDataset(hours=5, args=args), batch_size=args['batch_size'],
                             num_workers=16, shuffle=False, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


class Test_Dataset(Dataset):
    def __init__(self, num_samples) -> None:
        super().__init__()
        self.cleanlist = os.listdir('/my_datasets/vbd/clean_testset_wav/')
        self.noisylist = os.listdir('/my_datasets/vbd/noisy_testset_wav/')
        self.rirlist = torch.load('data_lists/rir.pt')

        self.num_samples = num_samples

        self.prob = {
            'reverb': 0.5,
            'noise': 0.5,
            'clipping': 0.5,
            'color': 0.5,
            'resample': 0.5,
        }

    def __len__(self):
        return self.num_samples

    def _decide_artifact(self):
        artifact = {}
        for name in self.prob.keys():
            artifact[name] = np.random.choice([False, True], p=[1 - self.prob[name], self.prob[name]])
        return artifact
    
    def _load_random_clean(self, cleanlist):
        clean_name = cleanlist[np.random.randint(len(cleanlist))]
        clean = torchaudio.load(os.path.join('../my_datasets/vbd/clean_testset_wav/', clean_name))[0][0]
        return clean
    
    def _load_random_noise(self, index):
        clean_name = self.cleanlist[index]
        clean = torchaudio.load(os.path.join('../my_datasets/vbd/clean_testset_wav/', clean_name))[0][0]
        noisy_name = self.noisylist[index]
        noisy = torchaudio.load(os.path.join('../my_datasets/vbd/noisy_testset_wav/', noisy_name))[0][0]
        noise = noisy - clean
        return noise
    
    def _load_random_rir(self, rirlist):
        while True:
            index = np.random.randint(len(rirlist))
            rir = torchaudio.load(rirlist[index])[0][0]
            rir = rir / torch.norm(rir, p=2)
            if torch.any(rir[:440]>0.5):
                break
        return rir
    
    def _add_noise(self, clean, noise, snr):
        clean_energy = torch.norm(clean, p=2) ** 2
        noise_energy = torch.norm(noise, p=2) ** 2
        A = torch.sqrt(clean_energy / (eps + noise_energy) / (10**(snr / 10)))
        mix = clean + A * noise
        return mix
    
    def _add_rir(self, clean, rir, early=False):
        if early:
            rir[480:] = 0
        mix = tools.convolve(clean, rir)[:len(clean)]
        return mix
    
    def _apply_codec(self, codec_effector, audio):
        return codec_effector.apply(audio.unsqueeze(-1), sample_rate=48000).squeeze(-1)
    
    def __getitem__(self, index):
        artifact = self._decide_artifact()
        clean = self._load_random_clean(self.cleanlist)

        index = np.random.randint(len(self.cleanlist))
        noise = self._load_random_noise(index)
        while len(noise) < len(clean):
            noise = torch.cat([noise, noise])
        noise = noise[:len(clean)]

        audio = clean
        if artifact['reverb']:
            rir = self._load_random_rir(self.rirlist)
            audio = self._add_rir(audio, rir, early=False)
            clean = self._add_rir(clean, rir, early=True)

        if artifact['noise']==True:
            snr = np.random.randint(-5, 10+1)
            audio = self._add_noise(audio, noise, snr)

        if artifact['color']:
            really_colored = False
            ###### lowpass ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['lowpass'] = True
                freq = np.random.randint(50, 6000)
                audio = torchaudio.functional.lowpass_biquad(audio, sample_rate=48000, cutoff_freq=freq)

            ###### lowshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['lowshelf'] = True
                freq = np.random.randint(50, 700)
                gain = np.random.randint(-12, 12)
                audio = torchaudio.functional.bass_biquad(audio, sample_rate=48000, gain=gain, central_freq=freq)

            ###### highpass ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['highpass'] = True
                freq = np.random.randint(10, 400)
                audio = torchaudio.functional.highpass_biquad(audio, sample_rate=48000, cutoff_freq=freq)

            ###### highshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['highshelf'] = True
                freq = np.random.randint(4500, 7500)
                gain = np.random.randint(-12, 12)
                audio = torchaudio.functional.treble_biquad(audio, sample_rate=48000, gain=gain, central_freq=freq)

            ###### peakshelf ######
            if np.random.uniform()<0.5:
                really_colored = True
                artifact['peakshelf'] = True
                num_peak = np.random.randint(1, 4+1)
                for _ in range(num_peak):
                    freq = np.random.randint(600, 7400)
                    gain = np.random.randint(-6, 6)
                    audio = torchaudio.functional.equalizer_biquad(audio, sample_rate=48000, center_freq=freq, gain=gain)
            
            if not really_colored:
                artifact['color'] = False
        
        if artifact['resample']: 
            ds = np.random.randint(2, 20+1) * 1000
            lowpass_order = 20
            length = len(audio)
            audio = torchaudio.functional.resample(audio, 48000, ds, lowpass_filter_width=lowpass_order)
            audio = torchaudio.functional.resample(audio, ds, 48000, lowpass_filter_width=lowpass_order)
            audio = audio[:length]
                
        if artifact['clipping']==True:
            threshold = np.random.uniform(0.1, 0.9)
            audio = torch.clamp(audio, -threshold, threshold)

        audio = torch.clamp(audio, -1, 1)

        audio = torchaudio.functional.resample(audio, 48000, 22050, lowpass_filter_width=20)
        clean = torchaudio.functional.resample(clean, 48000, 22050, lowpass_filter_width=20)

        return audio, clean


class Test_ExtremeLowSNR_Dataset(Test_Dataset):
    def __init__(self, num_samples) -> None:
        super().__init__(num_samples)
    
        self.prob = {
            'reverb': 0,
            'noise': 1,
            'clipping': 0,
            'color': 0,
            'resample': 0,
        }
    
    def __getitem__(self, index):
        artifact = self._decide_artifact()
        clean = self._load_random_clean(self.cleanlist)

        index = np.random.randint(len(self.cleanlist))
        noise = self._load_random_noise(index)
        while len(noise) < len(clean):
            noise = torch.cat([noise, noise])
        noise = noise[:len(clean)]

        audio = clean

        if artifact['noise']==True:
            snr = np.random.randint(-15, -5)
            audio = self._add_noise(audio, noise, snr)

        audio = torch.clamp(audio, -1, 1)

        audio = torchaudio.functional.resample(audio, 48000, 22050, lowpass_filter_width=20)
        clean = torchaudio.functional.resample(clean, 48000, 22050, lowpass_filter_width=20)

        return audio, clean


class Test_CodecDataset(Dataset):
    def __init__(self, num_samples, p_dict=None) -> None:
        super().__init__()
        self.cleanlist = torch.load('data_lists/vctk_wav48_silence_trimmed.pt')
        self.num_samples = num_samples
        self.p_dict = p_dict
        assert p_dict

        self.length = (300-1) * 480

        self.unseen_types = ['codec']

        self.codec_effectors = [
            AudioEffector(format="mp3", codec_config=CodecConfig(bit_rate=48000)),
            AudioEffector(format="mp3", codec_config=CodecConfig(bit_rate=96000)),
            AudioEffector(format="ogg", encoder="vorbis"),
            AudioEffector(format="ogg", encoder="opus"),
            AudioEffector(format="g722",),
            AudioEffector(format="mulaw",),
        ]

    def __len__(self):
        return self.num_samples
    
    def _load_random_clean(self, cleanlist):
        clean_name = cleanlist[np.random.randint(len(cleanlist))]
        clean = torchaudio.load(os.path.join('../my_datasets/vbd/clean_testset_wav/', clean_name))[0][0]
        return clean
    
    def _apply_codec(self, codec_effector, audio):
        return codec_effector.apply(audio.unsqueeze(-1), sample_rate=48000).squeeze(-1)
    
    def _decide_artifact(self):
        artifact = np.random.choice(self.unseen_types, p=[self.p_dict['codec'],] if self.p_dict else [1,])
        return artifact
    
    def __getitem__(self, index):
        artifact = self._decide_artifact()
        clean = self._load_random_clean(self.cleanlist)

        audio = clean.clone()
        
        if artifact == 'codec':
            i = np.random.randint(len(self.codec_effectors))
            audio = self._apply_codec(self.codec_effectors[i], audio)
        audio = torch.clamp(audio, -1, 1)

        audio = torchaudio.functional.resample(audio, 48000, 22050, lowpass_filter_width=20)
        clean = torchaudio.functional.resample(clean, 48000, 22050, lowpass_filter_width=20)

        return audio, clean