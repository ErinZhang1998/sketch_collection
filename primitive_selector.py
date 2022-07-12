import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn

import numpy as np 
import pickle

from collections import defaultdict
from torch.utils.data import Dataset

class HParams():
    def __init__(self):
        self.word_embed_dim = 128
        self.lstm_output_dim = 512
        self.lstm_layers = 2 
        self.lstm_drop_prob = 0.4
        self.num_primitives = 6
        self.vocab_size = None
        self.M = 2

def preprocess_dataset_language(path):
    f = open(path, "rb")
    data_raw = pickle.load(f)
    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    UNK = q2i["<unk>"]
    
    for _,info in data_raw.items():
        description = info['processed']
        [q2i[x] for x in description.lower().strip().split(" ")]
    return q2i

def collate_primitivedataset(seq_list):
    description_ts, primitive_types, affine_paramss = zip(*seq_list)
    lens = [len(x) for x in description_ts]
    seq_order = sorted(range(len(lens), key=lens.__getitem__, reverse=True))
    description_ts = [description_ts[i] for i in seq_order]
    primitive_types = torch.stack([primitive_types[i] for i in seq_order])
    affine_paramss = torch.stack([primitive_types[i] for i in seq_order])
    return description_ts, primitive_types, affine_paramss

class PrimitiveDataset(Dataset):
    def __init__(self, path, vocab, image_size=256.):
        super().__init__()
        self.path = path
        self.image_size = image_size
        
        f = open(self.path, "rb")
        self.data_raw = pickle.load(f)
        
        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.original_image_size = 256.

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, index):
        # Process language input 
        info = self.data_raw[index]
        description = info['processed']
        description_t = [self.vocab[x.lower()] for x in description.split(" ") if x.lower() in self.vocab_keys]
        description_t = torch.from_numpy(np.array(description_t)).long()
        
        # Process M (6,) and type
        primitive_type = torch.from_numpy(np.array([info['primitive_type']])).long()
        if 'M' in info:
            affine_params = info['M'].reshape(-1,)
            affine_params = torch.FloatTensor(affine_params)
        else:
            raise 
        return description_t, primitive_type, affine_params

class PrimitiveSelector(nn.Module):
    def __init__(self, hp):
        super().__init__()
        '''
        num_embeddings: vocab size 
        '''

        self.embed = nn.Embedding(hp.vocab_size, hp.word_embed_dim)
        self.lstm = nn.LSTM(
            input_size = hp.word_embed_dim, 
            hidden_size = hp.lstm_output_dim, 
            num_layers = hp.lstm_layers, 
            dropout = hp.lstm_drop_prob,
        )

        self.primitive_fc = nn.Linear(hp.lstm_output_dim, hp.num_primitives)
        # self.affine_fc = nn.Linear(hidden_embed_dim, 6) # l2 regression [x]
        self.affine_fc = nn.Linear(hp.lstm_output_dim, 2 * hp.M * 6)
        
    def forward(self, question): # question: PackedSequence 
        seq_tensor, seq_lengths = rnn.pad_packed_sequence(question, batch_first=True)               
        embedded_seq_tensor = self.embed(seq_tensor)
        seq_packed = rnn.pack_padded_sequence(np.transpose(embedded_seq_tensor,0,1), lens)
        seq_lstm, hidden = self.lstm(seq_packed, None)
        seq_last_layer = hidden[-1] # B x hidden_embed_dim
        
        prim_pred = self.primitive_fc(seq_last_layer)
        prim_param_pred = self.affine_fc(seq_last_layer)