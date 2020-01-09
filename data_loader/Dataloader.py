import sys
sys.path.append('../')

from torch.nn.utils.rnn import pack_sequence, pad_sequence
from data_loader import AudioData
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import util
import pickle
import numpy as np

def compute_mask(mixture, targets_list, mask_type):
    """
    Arguments:
        mixture: STFT of mixture signal(complex result) 
        targets_list: python list of target signal's STFT results(complex result)
        mask_type: ["irm", "ibm", "iam", "wfm", "psm"]
    """
    if mask_type == 'ibm':
        max_index = np.argmax(
            np.stack([np.abs(mat) for mat in targets_list]), 0)
        return np.array([max_index == s for s in range(len(targets_list))],dtype=np.float)

    if mask_type == "irm":
        denominator = sum([np.abs(mat) for mat in targets_list])
    else:
        if mask_type == "wfm":
            denominator = sum([np.power(np.abs(mat),2) for mat in targets_list])
        else:
            denominator = np.abs(mixture)
    if mask_type != "psm":
        if mask_type == "wfm":
            masks = [np.power(np.abs(mat),2) / denominator for mat in targets_list]
        else:
            masks = [np.abs(mat) / denominator for mat in targets_list]
    else:
        mixture_phase = np.angle(mixture)
        masks = [
            np.abs(mat) * np.cos(mixture_phase - np.angle(mat)) / denominator
            for mat in targets_list
        ]
    return np.array(masks)

class dataset(Dataset):
    def __init__(self, mix_reader, target_readers, cmvn_file='../cmvn.ark'):
        super(dataset).__init__()
        self.mix_reader = mix_reader
        self.target_readers = target_readers
        self.cmvn = pickle.load(open(cmvn_file, 'rb'))

    def __len__(self):
        return len(self.mix_reader)

    def __getitem__(self, index):
        if index >= len(self.mix_reader):
            raise ValueError
        mix_samp = self.mix_reader[index]
        T,F = mix_samp.shape[0],mix_samp.shape[1]
        target_lists = [target[index] for target in self.target_readers]
        wf = torch.from_numpy(compute_mask(mix_samp,target_lists,'wfm')).reshape(T*F,-1).type(torch.float32)
        ibm = torch.from_numpy(compute_mask(mix_samp,target_lists,'ibm')).reshape(T*F,-1).type(torch.float32)
        non_silent = torch.from_numpy(util.compute_non_silent(mix_samp)).reshape(T*F,-1).type(torch.float32)
        return torch.from_numpy(util.apply_cmvn(mix_samp,self.cmvn)).type(torch.float32), wf, ibm, non_silent
        





if __name__ == "__main__":
    mix_reader = AudioData(
        "/home/likai/data1/create_scp/tr_mix.scp", is_mag=True, is_log=True)
    target_readers = [AudioData("/home/likai/data1/create_scp/tr_s1.scp", is_mag=True, is_log=True),
                      AudioData("/home/likai/data1/create_scp/tr_s2.scp", is_mag=True, is_log=True)]
    dataset = dataset(mix_reader, target_readers)
    mix_samp, wf, ibm, non_silent = dataset[0]
    print(mix_samp.shape)
    print(wf.shape)
    print(ibm.shape)
    print(non_silent.shape)
