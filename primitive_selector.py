from re import L
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np 
import pickle

from collections import defaultdict
from torch.utils.data import Dataset

import wandb 
import argparse

class HParams():
    def __init__(self):
        self.word_embed_dim = 128
        self.lstm_output_dim = 512
        self.lstm_layers = 2 
        self.lstm_drop_prob = 0.4
        self.num_primitives = 5
        self.num_transformation_params = 6
        self.vocab_size = None
        self.M = 2
        self.weight_decay = 0.0
        
        self.save_every = 500
        self.print_every = 100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-enable_wandb", type=bool, default=False)
    parser.add_argument("-start_epoch", type=int, default=0)
    parser.add_argument("-num_epochs", type=int, default=50)
    parser.add_argument("-num_workers", type=int, default=5)
    parser.add_argument("-wandb_project_name", type=str, default="doodler-draw")
    parser.add_argument("-wandb_project_entity", type=str, default="erinz")
    parser.add_argument("-save_root_folder", type=str, default="/raid/xiaoyuz1/doodler_model_checkpoint")
    parser.add_argument("-train_file", type=str, required=True)
    parser.add_argument("-test_file", type=str, required=True)
    parser.add_argument("-vocab_file", type=str)
    
    args = parser.parse_args()
    return args

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
    affine_paramss = torch.stack([affine_paramss[i] for i in seq_order])
    
    # (N, 1) (N, num_transformation_params)
    return description_ts, primitive_types, affine_paramss

class PrimitiveDataset(Dataset):
    def __init__(self, path, vocab, num_transformation_params, image_size=256.):
        super().__init__()
        self.path = path
        self.image_size = image_size
        
        f = open(self.path, "rb")
        self.data_raw = pickle.load(f)
        
        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.original_image_size = 256.
        self.num_transformation_params = num_transformation_params

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, index):
        # Process language input 
        info = self.data_raw[index]
        description = info['processed']
        description_t = [self.vocab[x.lower()] for x in description.split(" ") if x.lower() in self.vocab_keys]
        description_t = torch.from_numpy(np.array(description_t)).long()
        
        # Process M (num_transformation_params,) and type
        primitive_type = torch.from_numpy(np.array([info['primitive_type']])).long()
        if 'M' in info:
            affine_params = info['M'].reshape(-1,)[:self.num_transformation_params]
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
        self.hp = hp
        self.embed = nn.Embedding(hp.vocab_size, hp.word_embed_dim)
        self.lstm = nn.LSTM(
            input_size = hp.word_embed_dim, 
            hidden_size = hp.lstm_output_dim, 
            num_layers = hp.lstm_layers, 
            dropout = hp.lstm_drop_prob,
        )

        self.primitive_fc = nn.Linear(hp.lstm_output_dim, hp.num_primitives)
        self.num_normal_param = 3
        self.affine_fc = nn.Linear(hp.lstm_output_dim, self.num_normal_param * hp.M * hp.num_transformation_params)
        
    def forward(self, question): # question: PackedSequence 
        seq_tensor, seq_lengths = rnn.pad_packed_sequence(question, batch_first=True)               
        embedded_seq_tensor = self.embed(seq_tensor)
        seq_packed = rnn.pack_padded_sequence(np.transpose(embedded_seq_tensor,0,1), seq_lengths)
        _, hidden = self.lstm(seq_packed, None)
        seq_last_layer = hidden[-1] # N x hidden_embed_dim
        
        prim_pred = self.primitive_fc(seq_last_layer) 
        prim_param_pred = self.affine_fc(seq_last_layer) # N x (self.num_normal_param * M * num_transformation_params)
        each_prim_param = torch.split(prim_param_pred, self.num_normal_param * self.hp.M, 1) # [N x (num_normal_param * M)]
        pi_list = [] # length num_transformation_params
        mu_list = []
        sigma_list = []
        for y in each_prim_param: # N x (num_normal_param * M)
            params = torch.split(y, self.num_normal_param, 1) # N x self.num_normal_param
            params_mixture = torch.stack(params) # M x N x self.num_normal_param
            pi, mu, sigma = torch.split(params_mixture, 1, 2) # M x N x 1
            pi = F.softmax(pi.transpose(0,1).squeeze(), dim=-1) # N x M
            mu = mu.transpose(0,1).squeeze().contiguous()
            sigma = torch.exp(sigma.transpose(0,1).squeeze())
            
            pi_list.append(pi)
            mu_list.append(mu)
            sigma_list.append(sigma)
        
        return prim_pred, pi_list, mu_list, sigma_list

class Trainer():
    def __init__(self, train_dataset, val_dataset, hp, args):
        
        self.hp = hp
        self.args = args 
        
        if args.enable_wandb:
            wandb.login()
            wandb.init(project=args.wandb_project_name, entity=args.wandb_project_entity, config=hp.__dict__)
        
        self.enable_wandb = args.enable_wandb and not wandb.run is None
        if self.enable_wandb:
            self.run_name = wandb.run.name 
        else:
            import datetime
            import time 
            ts = time.time()                                                                                            
            self.run_name = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') 
        self.save_folder = os.path.join(args.save_root_folder, self.run_name)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        
        self.train_dataset_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_primitivedataset)
        self.val_dataset_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_primitivedataset)

        self.device = "cuda" # if torch.cuda.is_available() else "cpu"
        self.model = PrimitiveSelector(hp).cuda()
        self.optimizer = optim.Adam(self._model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def make_target(self, affine_paramss):
        """create ground truth for training transformation parameters by stacking M copies of each parameter

        Args:
            affine_paramss: torch.Tensor (N, num_transformation_params)

        Returns:
            a torch.Tensor list of size num_transformation_params, one for each parameter. Each array has size (N, M)
        """
        return [
            torch.stack([affine_paramss[:,i]] * self.hp.M, 1) for i in range(affine_paramss.shape[1])
        ]
    
    def normal_pdf(self, x, mu, sigma):
        """Calculate univariate normal pdf for GMM

        Args:
            x : torch.Tensor (N, M) 
                ground truth parameter
            mu : torch.Tensor (N, M)
                predicted GMM means
            sigma : torch.Tensor (N, M)
                predicted GMM standard deviation

        Returns:
            pdf : torch.Tensor (N, M)
        """
        z = ( (x - mu) / sigma ) ** 2.0
        exp = torch.exp(-z / 2.0)
        norm = torch.sqrt(2.0 * np.pi) * sigma
        return exp / norm
    
    def log_losses(self, params_gt_list, pi_list, mu_list, sigma_list):
        """

        Args:
            params_gt_list : list of torch.Tensor (N, M)
                a list of GT transformation parameters
            pi_list : list of torch.Tensor (N, M)
                weights for combining the normal pdf in GMM
            mu_list : list of torch.Tensor (N, M)
                mean of GMM
            sigma_list : list of torch.Tensor (N, M)
                standard deviation of GMM
        Returns:
            losses : list of scalars
                each scalar is the log loss across the entire batch for one transformation parameter
        """
        losses = []
        for param,pi,mu,sigma in zip(params_gt_list, pi_list, mu_list, sigma_list):
            pdf = self.normal_pdf(param, mu, sigma)
            gmm_pdf = torch.sum(pi * pdf, 1)
            log_prob = torch.log(1e-5 + gmm_pdf) # (N,)
            loss = -torch.sum(log_prob)
            losses.append(loss)
        return losses
    
    def train(self):
        """Training script 
        """
        self.model.train()
        step = 0
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.num_epochs):
        
            for batch_idx, (description_ts, primitive_types, affine_paramss) in enumerate(self.train_dataset_loader):
                description_ts, primitive_types, affine_paramss = description_ts.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                params_gt_list = self.make_target(affine_paramss)
                
                prim_pred, pi_list, mu_list, sigma_list = self.model(description_ts)
                
                self.optimizer.zero_grad()
                cel = self.ce_loss(prim_pred, primitive_types)
                lls = self.log_losses(params_gt_list, pi_list, mu_list, sigma_list)
                total_ll = torch.sum(lls)
                total_lls = total_ll + cel
                
                wandb_dict = {'prim_type_loss' : cel.item(), 'total_param_loss' : total_ll.item()}
                for idx, ll in enumerate(lls):
                    wandb_dict[f'param_{idx}_loss'] = ll.item()
                wandb_dict['total_loss'] = total_lls.item()
                
                total_lls.backward()
                self.optimizer.step()
                
                if self.enable_wandb:
                    wandb.log(wandb_dict, step=step)
                else:
                    if step % self.hp.print_every == 0:
                        print_s = [f"Epoch {epoch} Iter {step}: "]
                        for k,v in wandb_dict.items():
                            print_s.append(f"{k} : {v}")
                        print(print_s.join(" | "))
                
                if step % self.hp.save_every == 0:
                    self.save_model(step)
                
                step += 1
                
    def save_model(self, step):
        
        torch_path_name = os.path.join(self.save_folder, f"{step}.pt")

        torch.save({
            'iteration' : step,
            'model_state_dict': self.model.state_dict(),
            'args': args,
        }, torch_path_name)

def main():
    args = get_args()
    hp = HParams()
    
    if args.vocab_file is None:
        vocab = preprocess_dataset_language(args.train_file)
    else:
        vocab = pickle.load(open(args.vocab_file, "rb"))
    hp.vocab_size = len(vocab)
    train_dataset = PrimitiveDataset(args.train_file, hp.num_transformation_params, vocab)
    val_dataset = PrimitiveDataset(args.test_file, hp.num_transformation_params, vocab)
    
    trainer = Trainer(train_dataset, val_dataset, hp, args)
    trainer.train()

if __name__ = "__main__":
    main()