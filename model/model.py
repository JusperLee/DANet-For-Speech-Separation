import sys
sys.path.append('../')

from config import option
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from . import torch_utils
import warnings
warnings.filterwarnings('ignore')

class DANet(nn.Module):
    def __init__(self, name='LSTM',
                 num_layer=2, input_size=129,
                 hidden_cells=600, emb_D=40,
                 dropout=0.0, bidirectional=True,
                 batch_first=True, activation="Tanh"):
        super(DANet, self).__init__()
        self.rnn = torch_utils.MultiRNN('LSTM', input_size, hidden_cells, 
                                           num_layers=num_layer, 
                                           bidirectional=bidirectional)
        self.linear = torch_utils.FCLayer(600, input_size*emb_D, nonlinearity='tanh')

        self.input_size = input_size
        self.emb_D = emb_D
        self.eps = 1e-8

    def forward(self, input_list):
        """
        input: the input feature; 
            shape: (B, T, F)

        ibm: the ideal binary mask used for calculating the 
            ideal attractors; 
            shape: (B, T*F, nspk)

        non_silent: the binary energy threshold matrix for masking 
            out T-F bins; 
            shape: (B, T*F, 1)
        """
        #self.rnn.flatten_parameters()
        input, ibm, non_silent, hidden = input_list
        B, T, F = input.shape
        # BT x H
        input, hidden = self.rnn(input,hidden)
        input = input.contiguous().view(-1, input.size(2))
        # BT x H -> BT x FD
        input = self.linear(input)
        # BT x FD -> B x TF x D
        v = input.view(-1, T*F, self.emb_D)
        # calculate the ideal attractors
        # first calculate the source assignment matrix Y
        # B x TF x nspk
        y = ibm*non_silent.expand_as(ibm)
        # attractors are the weighted average of the embeddings
        # calculated by V and Y
        # B x K x nspk
        v_y = torch.bmm(torch.transpose(v, 1, 2), y)
        # B x K x nspk
        sum_y = torch.sum(y, 1, keepdim=True).expand_as(v_y)
        # B x K x nspk
        attractor = v_y / (sum_y + self.eps)

        # calculate the distance bewteen embeddings and attractors
        # and generate the masks
        # B x TF x nspk
        dist = v.bmm(attractor)
        # B x TF x nspk
        mask = Fun.softmax(dist,dim=2)
        return mask, hidden
    
    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)

if __name__ == "__main__":
    opt = option.parse('../config/train.yml')
    net = DANet(**opt['DANet'])
    input = torch.randn(5,10,129)
    ibm = torch.randint(2, (5, 1290,2))
    non_silent = torch.randn(5, 1290, 1)
    out = net(input, ibm, non_silent)
