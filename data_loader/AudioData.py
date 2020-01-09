import sys
sys.path.append('../')

import utils.util as ut
from utils.stft_istft import STFT
import torch
import random
import numpy as np


class AudioData(object):
    '''
        Loading wave file
        scp_file: the scp file path
        other kwargs is stft's kwargs
        is_mag: if True, abs(stft)
    '''

    def __init__(self, scp_file, window='hann', nfft=256, window_length=256, hop_length=64, center=False, is_mag=True, is_log=True, chunk_size=32000, least=16000):
        self.wave = ut.read_scp(scp_file)
        self.wave_keys = [key for key in self.wave.keys()]
        self.STFT = STFT(window=window, nfft=nfft,
                         window_length=window_length, hop_length=hop_length, center=center)
        self.is_mag = is_mag
        self.is_log = is_log
        self.samp_list = []
        self.samp_stft = []
        self.chunk_size = chunk_size
        self.least = least
        self.split()
        self.stft()

    def __len__(self):
        return len(self.wave_keys)

    def split(self):
        for key in self.wave_keys:
            wave_path = self.wave[key]
            samp = ut.read_wav(wave_path)
            length = samp.shape[0]
            if length < self.least:
                continue
            if length < self.chunk_size:
                gap = self.chunk_size-length
                
                samp = np.pad(samp, (0, gap), mode='constant')
                self.samp_list.append(samp)
            else:
                random_start = 0
                while True:
                    if random_start+self.chunk_size > length:
                        break
                    self.samp_list.append(
                        samp[random_start:random_start+self.chunk_size])
                    random_start += self.least

    def stft(self):
        for samp in self.samp_list:
            self.samp_stft.append(self.STFT.stft(
                samp, is_mag=True, is_log=True))

    def __iter__(self):
        for stft in self.samp_stft:
            yield stft

    def __getitem__(self, index):
        return self.samp_stft[index]


if __name__ == "__main__":
    ad = AudioData("/home/likai/Desktop/create_scp/cv_mix.scp",
                   is_mag=True, is_log=True)
    print(ad.samp_stft[0].shape)
    print(ad.samp_stft[100].shape)
