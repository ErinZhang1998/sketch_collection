import read_datasets as rd
import constants as CONST
import numpy as np 
import pickle
from collections import defaultdict
import wandb 
wandb.login()
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

def create_split(arr_length):
    """Create train,dev,test split for a given sequence

    Parameters
    ----------
    arr_length : int 
        Length of the data to create split for

    Returns
    -------
    split_dict : dict
        A dictionary mapping data index to split, {"train", "dev", "test"} 
    """
    import random
    L = list(range(arr_length))
    split_dict = dict(zip(L, ["unassigned"] * arr_length))
    L.sort()  
    random.seed(1028)
    random.shuffle(L) 

    split_1 = int(0.8 * arr_length)
    split_2 = int(0.9 * arr_length)
    train_L = L[:split_1]
    split_dict.update(dict(zip(train_L, ["train"] * len(train_L))))
    dev_L = L[split_1:split_2]
    split_dict.update(dict(zip(dev_L, ["dev"] * len(dev_L))))
    test_L = L[split_2:]
    split_dict.update(dict(zip(test_L, ["test"] * len(test_L))))

    return split_dict

def prepare_data(df, templates, num_sampled_points = 200, use_projective = False):
    """Prepare and face and angel data for PrimitiveSelector training.

    Parameters
    ----------
    df : panda.DataFrame
        _description_
    templates : dict
        Mapping from template index to (template name, template numpy array (num_sampled_points, 2))
    num_sampled_points : int, optional
        Number of points to sample in each part, by default 200
    use_projective : bool, optional
        Whether to predict parameters for affine or projective transformation, by default False

    Returns
    -------
    all_data : list of dict
        List containing all the data to train PrimitiveSelector model.
    """
    import cv2
    from sklearn.metrics import mean_squared_error
    from tqdm import tqdm

    split_dict = create_split(len(df))
    all_data = []
    face_sketches = CONST.face_json['train_data']
    angel_sketches = CONST.angel_json['train_data']
    face_points = rd.process_all_to_part_convex_hull(face_sketches, list(CONST.face_parts_idx_dict_doodler.keys()), num_sampled_points)
    angel_points = rd.process_all_to_part_convex_hull(angel_sketches, list(CONST.angel_parts_idx_dict_doodler.keys()), num_sampled_points)
    
    for i in tqdm(range(len(df))):
        entry = df.iloc[i]
        if entry['category'] == 'face':
            raw_desc = "{} {}".format(entry['text_1'], CONST.face_parts_idx_dict_doodler[entry['part']])
            desc = "{} {}".format(entry['no_punc_str_1'], CONST.face_parts_idx_dict_doodler[entry['part']])
            sketch_data = face_points[entry['image_1']]
        else:
            raw_desc = "{} {}".format(entry['text_1'], CONST.angel_parts_idx_dict_doodler[entry['part']])
            desc = "{} {}".format(entry['no_punc_str_1'], CONST.angel_parts_idx_dict_doodler[entry['part']])
            sketch_data = angel_points[entry['image_1']]
        
        min_template_squared_error = np.inf
        min_M = None
        min_template_idx = None
        for template_idx, (_, template) in templates.items():
            data = sketch_data[entry['part']]
            M = rd.get_transform(template, data, projective=use_projective)
            if use_projective:
                result = cv2.perspectiveTransform(template, M).reshape(-1,2)
            else:
                result = cv2.transform(np.array([template], copy=True).astype(np.float32), M)[0][:,:-1]
            squared_error = np.sum(mean_squared_error(result, data, multioutput='raw_values'))
            if squared_error < min_template_squared_error:
                min_template_idx = template_idx
                min_template_squared_error = squared_error
                min_M = M.reshape(-1,)
        transform_mat = min_M.reshape(3,3)
        theta, scale_mat, shear_mat = rd.decompose_affine(transform_mat)
        
        
        info = {
            'category' : entry['category'],
            'image_idx' : entry['image_1'],
            'part' : entry['part'],
            'raw' : raw_desc,
            'processed' : desc,
            'primitive_type' : int(min_template_idx),
            'M' : min_M,
            'error' : min_template_squared_error,
            'split' : split_dict[i],
        }
        info["theta"] = theta
        info["sx"] = scale_mat[0][0]
        info["sy"] = scale_mat[1][1]
        info["hx"] = shear_mat[0][1]
        info["tx"] = transform_mat[0][2]
        info["ty"] = transform_mat[1][2]
        T_rotate = rd.get_rotation_matrix(theta)
        T_scale = rd.get_scale_matrix(info["sx"], info["sy"])
        T_shear = rd.get_shear_matrix(info["hx"], 0)
        T_translate = T_translate = rd.get_translation_matrix(info["tx"], info["ty"])
        T = T_translate @ T_rotate @ T_scale @ T_shear

        if not rd.check_close(info["M"].reshape(-1,), T.reshape(-1,)):
            print(i)
        all_data.append(info)
    return all_data

def preprocess_dataset_language(path):
    f = open(path, "rb")
    data_raw = pickle.load(f)
    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    UNK = q2i["<unk>"]
    
    for info in data_raw:
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
        """Create ground truth for training transformation parameters by stacking M copies of each parameter

        Parameters
        ----------
        affine_paramss : torch.Tensor
            (N, num_transformation_params)

        Returns
        -------
        list of torch.Tensor
            GT for calculating log likelihood loss, each has shape (N, M)
            list of size num_transformation_params
        """
        return [
            torch.stack([affine_paramss[:,i]] * self.hp.M, 1) for i in range(affine_paramss.shape[1])
        ]
    
    def normal_pdf(self, x, mu, sigma):
        """Calculate univariate normal pdf for GMM

        Parameters
        ----------
        x : torch.Tensor
            (N, M)
        mu : torch.Tensor
            (N, M)
            predicted GMM means
        sigma : torch.Tensor
            (N, M)
            predicted GMM standard deviation

        Returns
        -------
        pdf : torch.Tensor
            (N, M)
            predicted probability
        """
        z = ( (x - mu) / sigma ) ** 2.0
        exp = torch.exp(-z / 2.0)
        norm = torch.sqrt(2.0 * np.pi) * sigma
        pdf = exp / norm
        return pdf
    
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

if __name__ == "__main__":
    main()