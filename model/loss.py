import sys
sys.path.append('../')

import torch
from utils import util



class Loss(object):
    '''
    """
    MSE as the training objective. The mask estimation loss is calculated.
    You can also change it into the spectrogram estimation loss, which is 
    to calculate the MSE between the clean source spectrograms and the 
    masked mixture spectrograms.

    mix_samp: the spectrogram of the mixture;
        shape: (B, T, F)

    wf: the target masks, which are the wiener-filter like masks here;
        shape: (B, T*F, nspk)

    mask: the estimated masks generated by the network;
        shape: (B, T*F, nspk)
    """
    '''

    def __init__(self, mix_samp, wf, mask):
        self.mix_samp = mix_samp
        self.wf = wf
        self.mask = mask

    def loss(self):
        #loss = self.mix_samp.view(-1, self.mix_samp.size(1)*self.mix_samp.size(2), 1).expand(
        #    self.mix_samp.size(0), self.mix_samp.size(1), self.wf.size(2)) * (self.wf - self.mask)
        loss = self.wf-self.mask
        loss = loss.view(-1, loss.size(1)*loss.size(2))
        return torch.mean(torch.sum(torch.pow(loss, 2), 1))
