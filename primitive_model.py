from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical

class ConvBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            nn.ReLU(), #n.LeakyReLU(0.2, inplace=True)
            # nn.Conv2d(filters, filters, 3, padding=1),
            # nn.ReLU(), #n.LeakyReLU(0.2, inplace=True)
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def zscore_canvases(x):
    return (x - 0.0243)/0.1383


class CNN_Encoder(nn.Module):
    def __init__(self, input_channel, filters = [16, 32, 64, 128, 512]):
        super().__init__()
        filters = [input_channel] + filters
        chan_in_out = list(zip(filters[0:-1], filters[1:]))
        conv_blocks = []
        for ind,(chan_in, chan_out) in enumerate(chan_in_out):
            is_not_last = ind < (len(chan_in_out) - 1)
            block = ConvBlock(chan_in, chan_out, downsample=is_not_last)
            conv_blocks.append(block)
        self.encoder = nn.ModuleList(conv_blocks)
        #nn.Tanh()
    
    def forward(self, x):
        x = zscore_canvases(x)
        x = self.encoder(x) # (N, image_output_dim, 4 x 4)
        x = x.permute(0,2,3,1) 
        x = x.view(x.size(0),-1,x.size(-1)) 
        return x

class Combine(nn.Module):
    def __init__(self, lstm_output_dim, image_output_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lstm_output_dim + image_output_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, v, q):
        """
        :param v: (N, image_output_dim)
        :param q: (N, lstm_output_dim)
        :return:
        """
        vq = torch.cat([v,q], dim=1)
        vq = self.net(vq)

        return vq

class CoAttention(nn.Module):
    def __init__(self, lstm_output_dim, image_output_dim, coatt_hidden_dim):
        super().__init__()
        self.W_V_to_Q_dim = nn.Parameter(torch.randn(lstm_output_dim, image_output_dim))
        self.W_v = nn.Parameter(torch.randn(coatt_hidden_dim, image_output_dim))
        self.W_q = nn.Parameter(torch.randn(coatt_hidden_dim, lstm_output_dim))
        self.w_hv = nn.Parameter(torch.randn(coatt_hidden_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(coatt_hidden_dim, 1))
    
    def forward(self, V, Q):
        """
        :param V: (N, image_output_dim, I)
        :param Q: (N, L, lstm_output_dim)
        :return:
            v: (N, image_output_dim)
            q: (N, lstm_output_dim)
        """
        C = torch.matmul(Q, torch.matmul(self.W_V_to_Q_dim, V)) # N x L x I 
        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C)) # N x coatt_hidden_dim x I
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1))) # N x coatt_hidden_dim x L
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) 
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) 
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) 
        q = torch.squeeze(torch.matmul(a_q, Q))                  

        return v, q

class PrimitiveSelector(nn.Module):
    def __init__(self, hp, img_enc = None):
        super().__init__()
        self.hp = hp
        self.embed = nn.Embedding(hp.vocab_size, hp.word_embed_dim)
        self.lstm = nn.LSTM(
            input_size = hp.word_embed_dim, 
            hidden_size = hp.lstm_output_dim, 
            num_layers = hp.lstm_layers, 
            dropout = hp.lstm_drop_prob,
        )
        
        last_dim = hp.lstm_output_dim if img_enc is None else hp.combined_dim
        self.primitive_fc = nn.Linear(last_dim, hp.num_primitives)
        self.gmm_network = nn.Linear(last_dim, len(hp.parameter_names) * 2 * hp.M)
        self.pi_network = nn.Linear(last_dim, len(hp.parameter_names) * hp.M)
        
        if img_enc is not None:
            self.img_enc = img_enc
            self.coatt = CoAttention(hp.lstm_output_dim, hp.image_output_dim, hp.coatt_hidden_dim)
            self.combine_net = Combine(hp.lstm_output_dim, hp.image_output_dim, hp.combined_dim)

    def unpack_output(self, params_list, pis_list):
        normal_dists, pi_dists = [],[]
        for i in range(len(self.hp.parameter_names)):
            param = params_list[i]
            pi = pis_list[i]
            mean, sd = torch.split(param, param.shape[1] // 2, dim=1) # each: N x M
            mean = torch.stack(mean.split(mean.shape[1] // self.hp.M, 1)) # stack N x 1 --> M x N x 1
            sd = torch.stack(sd.split(sd.shape[1] // self.hp.M, 1)) # M x N x 1
            normal_dist = Normal(mean.transpose(0, 1), (F.elu(sd)+1+1e-7).transpose(0, 1)) # N x M x 1
            pi_dist = OneHotCategorical(logits=pi)

            normal_dists.append(normal_dist)
            pi_dists.append(pi_dist)
        return normal_dists, pi_dists
    
    def forward(self, question, image=None):
        """
        :param question: packed sequence 
        """
        seq_tensor, seq_lengths = rnn.pad_packed_sequence(question, batch_first=True) # seq_tensor: (N, L)             
        embedded_seq_tensor = self.embed(seq_tensor)
        seq_packed = rnn.pack_padded_sequence(
            torch.transpose(embedded_seq_tensor,0,1), 
            seq_lengths)
        h, (hidden,_) = self.lstm(seq_packed, None) # h is L x N x lstm_output_dim
        
        if image is not None:
            img_feat = self.img_enc(image) # (N, image_output_dim, I)
            h = torch.transpose(h, 0, 1) # N x L x lstm_output_dim
            v,q = self.coatt(img_feat, h)
            y = self.combine_net(v,q)
        else: 
            y = hidden[-1] # N x lstm_output_dim
        
        prim_pred = self.primitive_fc(y) 
        params = self.gmm_network(y)
        pis = self.pi_network(y)
        
        params_list = torch.split(params, 2 * 1 * self.hp.M, dim=1) # each: N x 2 * M
        pis_list = torch.split(pis, self.hp.M, dim=1) # each: N x M
        normal_dists, pi_dists = self.unpack_output(params_list, pis_list)
        return prim_pred, normal_dists, pi_dists
