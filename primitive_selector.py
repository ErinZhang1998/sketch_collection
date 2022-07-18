from cmath import isclose
import sys    
path_to_module = '/home/xiaoyuz1/sketch_collection'
sys.path.append(path_to_module)

import pandas as pd
import read_datasets as rd
import numpy as np 
import pickle
from collections import defaultdict
import wandb 
import argparse
import cv2 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import io
import PIL
from skimage.metrics import structural_similarity 

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
import json

class Constants():
    def __init__(self):

        self.face_parts_idx = [0,1,2,4,6]
        self.face_parts_idx_dict = {
            0: "eyes",
            1: "nose",
            2: "mouth",
            3: "ear",
            4: "hair",
            5: "moustache",
            6: "outline of face",
        }
        self.face_parts_idx_dict_doodler = {
            0: "eyes",
            1: "nose",
            2: "mouth",
            3: "ear",
            4: "hair",
            5: "moustache",
            6: "face",
        }
        self.face_json = json.load(open('/raid/xiaoyuz1/PG/face.ndjson', 'r'))

        self.angel_json = json.load(open('/raid/xiaoyuz1/PG/angel.ndjson', 'r'))

        self.angel_parts_idx = [0,1,2,3,4,5,7]
        self.angel_parts_idx_dict = {
            0:"halo",1 : "eyes",2:"nose",3:"mouth",4:"face",5:"body", 6: "limbs", 7:"wings",
        }

class HParams():
    def __init__(self):
        self.word_embed_dim = 64
        self.lstm_output_dim = 512
        self.lstm_layers = 2 
        self.lstm_drop_prob = 0.4
        self.num_primitives = 5
        self.parameter_names = {0:"theta", 1:"sx", 2:"sy", 3:"hx", 4:"tx", 5:"ty"}
        self.num_transformation_params = 6
        self.vocab_size = None
        self.M = 3
        self.weight_decay = 0.0
        self.batch_size = 64
        self.lr = 0.001

class CommandParams():
    def __init__(self):
        self.enable_wandb = False 
        self.start_epoch = 0
        self.num_epochs = 50
        self.num_workers = 0
        self.wandb_project_name = "doodler-draw"
        self.wandb_project_entity = "erinz"
        self.save_root_folder = "/raid/xiaoyuz1/doodler_model_checkpoint"
        self.train_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train.pkl"
        self.dev_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_val.pkl"
        self.test_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_test.pkl"
        self.train_seq_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train_sequences.pkl"
        self.dev_seq_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_val_sequences.pkl"
        self.test_seq_file = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_test_sequences.pkl"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-enable_wandb", action='store_true')
    parser.add_argument("-start_epoch", type=int, default=0)
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-wandb_project_name", type=str, default="doodler-draw")
    parser.add_argument("-wandb_project_entity", type=str, default="erinz")
    parser.add_argument("-save_root_folder", type=str, default="/raid/xiaoyuz1/doodler_model_checkpoint")
    parser.add_argument("-train_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train.pkl")
    parser.add_argument("-dev_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_dev.pkl")
    parser.add_argument("-test_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_test.pkl")
    parser.add_argument("-train_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train_sequences.pkl")
    parser.add_argument("-dev_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_dev_sequences.pkl")
    parser.add_argument("-test_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_test_sequences.pkl")
    
    parser.add_argument("-word_embed_dim", type=int, default=64)
    parser.add_argument("-lstm_output_dim", type=int, default=512)
    parser.add_argument("-lstm_layers", type=int, default=2)
    parser.add_argument("-lstm_drop_prob", type=float, default=0.2)
    parser.add_argument("-num_primitives", type=int, default=5)
    parser.add_argument("-parameter_names", nargs='+', default=["theta","sx","sy","hx","tx","ty"])
    parser.add_argument("-num_transformation_params", type=int, default=6)
    parser.add_argument("-vocab_size", type=int)
    parser.add_argument("-M", type=int, default=2)
    parser.add_argument("-weight_decay", type=float, default=0.0)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.0001)

    parser.add_argument("-save_every", type=int, default=1000)
    parser.add_argument("-print_every", type=int, default=500)
    parser.add_argument("-num_visualize", type=int, default=9)
    parser.add_argument("-num_sampled_points", type=int, default=200)
    parser.add_argument("-use_projective", action="store_true")
    parser.add_argument("-canvas_size", type=float, default=256.0)

    args = parser.parse_args()
    
    return args

def rendered_image_to_01(img):
    img[img < 1] = 1
    img[img == 255] = 0
    return img


def get_affine_transformation(info):
    T_rotate = rd.get_rotation_matrix(info["theta"])
    T_scale = rd.get_scale_matrix(info["sx"], info["sy"])
    T_shear = rd.get_shear_matrix(info["hx"], 0)
    T_translate = T_translate = rd.get_translation_matrix(info["tx"], info["ty"])
    T = T_translate @ T_rotate @ T_scale @ T_shear
    return T

def preprocess_dataset_language(path):
    """Return map from word to word index.

    Parameters
    ----------
    path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    data_raw = pickle.load(open(path, "rb"))
    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    UNK = q2i["<unk>"]
    
    for info in data_raw:
        description = info['processed']
        [q2i[x] for x in description.lower().strip().split(" ")]
    return q2i

def preprocess_sequence(path, templates, num_sampled_points, use_projective):
    """Original sequence (procssed with convex hull), template sequence, fitted templated sequence

    Parameters
    ----------
    path : _type_
        _description_
    num_sampled_points : _type_
        _description_
    use_projective : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    CONST = Constants()
    data_raw = pickle.load(open(path, "rb"))
    all_sequences = []
    for info in data_raw:
        if info['category'] == 'face':
            drawing_raw = CONST.face_json['train_data'][info['image_idx']]
            part_indices = list(CONST.face_parts_idx_dict_doodler.keys())
        else:
            drawing_raw = CONST.angel_json['train_data'][info['image_idx']]
            part_indices = list(CONST.angel_parts_idx_dict_doodler.keys())

        parts = rd.process_quickdraw_to_part_convex_hull(
            drawing_raw,
            part_indices,
            b_spline_num_sampled_points=num_sampled_points,
        )
        data = parts[info['part']]
        template = templates[info['primitive_type']][1]
        transform_mat = np.asarray(info['M']).astype(float).reshape(3,3)
        
        if use_projective:
            result = cv2.perspectiveTransform(template, transform_mat).reshape(-1,2)
        else:
            result = cv2.transform(np.array([template]).astype(
                np.float32), transform_mat)[0][:,:-1]
        
        all_sequences.append([data, template, result])
    return all_sequences

def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def collate_primitivedataset(seq_list):
    description_ts, primitive_types, affine_paramss, indices = zip(*seq_list)
    lens = [len(x) for x in description_ts]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    description_ts = [description_ts[i] for i in seq_order]
    
    primitive_types = torch.stack([primitive_types[i] for i in seq_order])
    affine_paramss = torch.stack([affine_paramss[i] for i in seq_order])
    indices = torch.stack([indices[i] for i in seq_order])
    
    return description_ts, primitive_types, affine_paramss, indices

class PrimitiveDataset(Dataset):
    def __init__(self, path, vocab, num_transformation_params):
        super().__init__()
        self.path = path        
        self.data_raw = pickle.load(open(self.path, "rb"))
        # self.df = pd.DataFrame(self.data_raw)

        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.vocab_reverse = dict(zip(vocab.values(), vocab.keys()))
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
        primitive_type = torch.tensor(info['primitive_type']).long()
        
        affine_params = torch.FloatTensor(np.array([
            info["theta"],info["sx"],info["sy"],info["hx"],info["tx"],info["ty"],
        ]))
        return description_t, primitive_type, affine_params, torch.tensor(index).long()


class PrimitiveSelector(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.embed = nn.Embedding(hp.vocab_size, hp.word_embed_dim)
        self.lstm = nn.LSTM(
            input_size = hp.word_embed_dim, 
            hidden_size = hp.lstm_output_dim, 
            num_layers = hp.lstm_layers, 
            dropout = hp.lstm_drop_prob,
        )

        self.primitive_fc = nn.Linear(hp.lstm_output_dim, hp.num_primitives)
        self.gmm_network = nn.Linear(hp.lstm_output_dim, hp.num_transformation_params * 2 * 1 * hp.M)
        self.pi_network = nn.Linear(hp.lstm_output_dim, hp.num_transformation_params * hp.M)

    def forward(self, question):
        seq_tensor, seq_lengths = rnn.pad_packed_sequence(question, batch_first=True)               
        embedded_seq_tensor = self.embed(seq_tensor)
        seq_packed = rnn.pack_padded_sequence(
            torch.transpose(embedded_seq_tensor,0,1), 
            seq_lengths)
        _, (hidden,_) = self.lstm(seq_packed, None)
        seq_last_layer = hidden[-1] # N x lstm_output_dim
        prim_pred = self.primitive_fc(seq_last_layer) 

        params = self.gmm_network(seq_last_layer)
        pis = self.pi_network(seq_last_layer)
        # mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        # mean = torch.stack(mean.split(mean.shape[1] // self.hp.M, 1))
        # sd = torch.stack(sd.split(sd.shape[1] // self.hp.M, 1))
        # normal_dist = Normal(mean.transpose(0, 1), (F.elu(sd)+1+1e-7).transpose(0, 1))
        # pi_dist = OneHotCategorical(logits=pis)

        params_list = torch.split(params, 2 * 1 * self.hp.M, dim=1) # each: N x 2 * M
        pis_list = torch.split(pis, self.hp.M, dim=1) # each: N x M
        
        normal_dists, pi_dists = [],[]
        for i in range(self.hp.num_transformation_params):
            param = params_list[i]
            pi = pis_list[i]
            mean, sd = torch.split(param, param.shape[1] // 2, dim=1) # each: N x M
            mean = torch.stack(mean.split(mean.shape[1] // self.hp.M, 1)) # stack N x 1 --> M x N x 1
            sd = torch.stack(sd.split(sd.shape[1] // self.hp.M, 1)) # M x N x 1
            normal_dist = Normal(mean.transpose(0, 1), (F.elu(sd)+1+1e-7).transpose(0, 1)) # N x M x 1
            pi_dist = OneHotCategorical(logits=pi)

            normal_dists.append(normal_dist)
            pi_dists.append(pi_dist)

        return prim_pred, normal_dists, pi_dists

class Meter(object):
    def __init__(self, meter_name):
        self.meter_name = meter_name
        self.count = 0
        self.correct_count = 0
        self.metric_dict = defaultdict(float)
    
    def reset(self):
        self.count = 0
        self.correct_count = 0
        self.metric_dict = defaultdict(float)
    
    # def log_loss(self, ll_list):
        
    
    def log_metric(self, pred_type, gt_type, gt_img, pred_img):
        if pred_type == gt_type:
            self.correct_count += 1 
        self.count += 1
        mse = np.mean((gt_img.astype(np.float32) - pred_img.astype(np.float32)) ** 2)
        psnr = 100 if np.isclose(mse, 0.0, rtol=1.0, atol=1e-5) else 20 * np.log10(255 / (np.sqrt(mse)))
        ssim = structural_similarity(gt_img, pred_img, multichannel=False, data_range=255)
        new_mse = self.metric_dict["mse"] + (mse - self.metric_dict["mse"]) / self.count
        self.metric_dict["mse"] = new_mse 
        new_psnr = self.metric_dict["psnr"] + (psnr - self.metric_dict["psnr"]) / self.count
        self.metric_dict["psnr"] = new_psnr 
        new_ssim = self.metric_dict["ssim"] + (ssim - self.metric_dict["ssim"]) / self.count
        self.metric_dict["ssim"] = new_ssim 
    
    def finalize_metric(self):
        metric_print_dict = {
            f"{self.meter_name}_acc" : self.correct_count / self.count
        }
        for k,v in self.metric_dict.items():
            metric_print_dict[f"{self.meter_name}_{k}"] = v
        return metric_print_dict
        
def print_dict(pd, print_s = []):
    for k,v in pd.items():
        print_s.append(f"{k} : {v}")
    print("\n\t".join(print_s))       

class Trainer():
    def __init__(self, hp, args, vocab, templates):
        
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
        self.templates = templates
        self.train_dataset = PrimitiveDataset(args.train_file, vocab, hp.num_transformation_params)
        self.test_dataset = PrimitiveDataset(args.test_file, vocab, hp.num_transformation_params)
        self.train_dataset_loader = DataLoader(
            self.train_dataset, 
            batch_size=hp.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            collate_fn=collate_primitivedataset)
        self.test_dataset_loader = DataLoader(
                self.test_dataset, 
                batch_size=hp.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers, 
                collate_fn=collate_primitivedataset)
        self.train_sequences = pickle.load(open(args.train_seq_file, 'rb'))
        self.test_sequences = pickle.load(open(args.test_seq_file, 'rb'))

        self.device = "cuda" # if torch.cuda.is_available() else "cpu"
        self.model = PrimitiveSelector(hp).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.test_meter = Meter("test")
        self.train_meter = Meter("train")
        
        self.num_pngs_per_row = 3
        num_rows = args.num_visualize // self.num_pngs_per_row
        if num_rows * self.num_pngs_per_row < args.num_visualize:
            num_rows += 1
        self.num_rows = num_rows
    
    def loss(self, y_list, normal_dists, pi_dists, calculate_mean=True):
        losses = []
        for param_idx,(y,normal_dist,pi_dist) in enumerate(zip(y_list, normal_dists, pi_dists)):
            ys = y.unsqueeze(1).expand_as(normal_dist.loc)
            loglik = normal_dist.log_prob(ys)
            loglik = torch.sum(loglik, dim=2)
            loss = -torch.logsumexp(pi_dist.logits + loglik, dim=1)
            if calculate_mean:
                loss_mean = loss.mean()
                losses.append(loss_mean)
            else:
                losses.append(loss)
            # if loss_mean < 0:
            #     print(f"{param_idx}: ", torch.exp(loglik))
        return losses
    
    def calculate_metric(self, descriptions, dataset_indices, prim_types, param_samples, plot_indices=None, train=False):
        """_summary_

        Parameters
        ----------
        dataset_indices : _type_
            _description_
        prim_types : torch.Tensor (N,)
            _description_
        param_samples : list of torch.Tensor
            list of tensors of shape (N, output_dim) 

        Returns
        -------
        """
        plot_i = 0
        dataset_indices_plot = []
        if not train:
            fig = plt.figure(figsize=(self.num_pngs_per_row * 5, self.num_rows * 5)) 
            fig.patch.set_alpha(1)  
        for idx, data_idx in enumerate(dataset_indices):
            data_idx = data_idx.item()
            prim_type = prim_types[idx].item()
            if train:
                info = self.train_dataset.data_raw[data_idx]
            else:
                info = self.test_dataset.data_raw[data_idx]
           
            pred_template_name, pred_template = self.templates[int(prim_type)]
            if train:
                data, template, result, gt_img = self.train_sequences[data_idx]
            else:
                data, template, result, gt_img = self.test_sequences[data_idx]
            pred_info = {}
            pred_params = [sample[idx].item() for sample in param_samples]
            pred_info["theta"],pred_info["sx"],pred_info["sy"],pred_info["hx"],pred_info["tx"],pred_info["ty"] = pred_params
            pred_M = get_affine_transformation(pred_info)
            if self.args.use_projective:
                pred_template_pred_param = cv2.perspectiveTransform(pred_template, pred_M).reshape(-1,2)
            else:
                pred_template_pred_param = cv2.transform(np.array([pred_template]).astype(np.float32), pred_M)[0][:,:-1]

            # gt_img = np.asarray(rd.render_img([result], line_diameter=3))
            pred_img = np.asarray(rd.render_img([pred_template_pred_param], line_diameter=3))
            if train:
                self.train_meter.log_metric(prim_type, int(info["primitive_type"]), gt_img, pred_img)
            else:
                self.test_meter.log_metric(prim_type, int(info["primitive_type"]), gt_img, pred_img)
            
            if not train and idx in plot_indices:
                dataset_indices_plot.append(data_idx)
                gt_template_name,_ = self.templates[int(info["primitive_type"])]
                ax = plt.subplot(self.num_rows, self.num_pngs_per_row, plot_i+1)
                
                if self.args.use_projective:
                    correct_template_pred_param = cv2.perspectiveTransform(template, pred_M).reshape(-1,2)
                else:
                    correct_template_pred_param = cv2.transform(np.array([template]).astype(np.float32), pred_M)[0][:,:-1]
                if self.args.use_projective:
                    pred_template_pred_param = cv2.perspectiveTransform(pred_template, pred_M).reshape(-1,2)
                else:
                    pred_template_pred_param = cv2.transform(np.array([pred_template]).astype(np.float32), pred_M)[0][:,:-1]

                ax.scatter(data[:,0], data[:,1], s=1, c='b')
                ax.scatter(result[:,0], result[:,1], s=1, alpha=0.5, c='r')
                ax.scatter(correct_template_pred_param[:,0], correct_template_pred_param[:,1], s=1, c='darkred', alpha=0.5)
                ax.scatter(pred_template_pred_param[:,0], pred_template_pred_param[:,1], s=1, c='lime')
                plt.xlim(-self.args.canvas_size,self.args.canvas_size)
                plt.ylim(self.args.canvas_size,-self.args.canvas_size)
                gt_params = [info["theta"],info["sx"],info["sy"],info["hx"],info["tx"],info["ty"]]
                title = f"{gt_template_name},{pred_template_name}\n" 
                desc = info["processed"]
                desc2 = " ".join([self.test_dataset.vocab_reverse[j.item()] for j in descriptions[idx]])
                title += f"{desc},{desc2}\n"
                
                for pi,(p1,p2) in enumerate(zip(gt_params, pred_params)):
                    if pi == 0:
                        p1 = np.degrees(p1)
                        p2 = np.degrees(p2)
                    title += f"{self.hp.parameter_names[pi]}: {p1:.3f} {p2:.3f}"
                    if pi % 2 == 0:
                        title += "\n"
                    else:
                        title += " | "
                ax.set_title(title, y=0.7)
                plot_i += 1
        if not train:
            return fig, dataset_indices_plot
       
    def train(self):
        self.model.train()
        step = 0
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.num_epochs):
        
            for batch_idx, (description_ts, primitive_types, affine_paramss, dataset_indices) in enumerate(self.train_dataset_loader):
                description_ts_packed = rnn.pack_sequence(description_ts)
                
                description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                # params_gt_list = self.make_target(affine_paramss)
                params_gt_list = [
                    affine_paramss[:,i].view(-1,1) for i in range(affine_paramss.shape[1])
                ]

                # prim_pred, pi_list, mu_list, sigma_list = self.model(description_ts_packed)
                prim_pred, normal_dists, pi_dists = self.model(description_ts_packed)
                # prim_types = torch.argmax(prim_pred, dim=1)
                # param_samples = self.sample_parameters(normal_dists, pi_dists) # each: N x 1                
                self.optimizer.zero_grad()
                
                cel = self.ce_loss(prim_pred, primitive_types)
                # lls = self.log_losses(params_gt_list, pi_list, mu_list, sigma_list, epoch)
                lls = self.loss(params_gt_list, normal_dists, pi_dists)
                total_ll = torch.stack(lls).sum()
                total_lls = total_ll + cel
                
                wandb_dict = {'prim_type_loss' : cel.item(), 'total_param_loss' : total_ll.item()}
                for idx, ll in enumerate(lls):
                    wandb_dict[f'{self.hp.parameter_names[idx]}_loss'] = ll.item()
                wandb_dict['total_loss'] = total_lls.item()
                
                total_lls.backward()
                self.optimizer.step()
                
                if self.enable_wandb:
                    wandb.log(wandb_dict, step=step)
                
                if step % self.args.print_every == 0:
                    print_s = [f"Epoch {epoch} Iter {step}: "]
                    print_dict(wandb_dict, print_s)
                
                if step % self.args.save_every == 1:
                    self.model.eval()
                    with torch.no_grad(): 
                        for batch_idx, (description_ts, primitive_types, affine_paramss, dataset_indices) in enumerate(self.train_dataset_loader):
                            description_ts_packed = rnn.pack_sequence(description_ts)
                            
                            description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                            prim_types = torch.argmax(prim_pred, dim=1)
                            param_samples = self.sample_parameters(normal_dists, pi_dists) # each: N x 1   
                            _ = self.calculate_metric(description_ts, dataset_indices, prim_types, param_samples, plot_indices=None, train=True)
                    log_dict = self.train_meter.finalize_metric()
                    if self.enable_wandb:
                        wandb.log(log_dict, step=step)
                    print_s = [f"Train @ Iter {step}: "]
                    print_dict(log_dict, print_s)
                    self.train_meter.reset()
                    
                    self.evaluate(step)
                    self.save_model(step)
                    
                    self.model.train()
                step += 1
    
    def sample_parameters(self, normal_dists, pi_dists):
        samples = []
        for param_idx,(normal_dist, pi_dist) in enumerate(zip(normal_dists, pi_dists)):
            sample = torch.sum(pi_dist.sample().unsqueeze(2) * normal_dist.sample(), dim=1)
            samples.append(sample)
        return samples

    def evaluate(self, step):
        
        self.test_meter.reset()
        with torch.no_grad(): 
            for batch_idx, (description_ts, primitive_types, affine_paramss, dataset_indices) in enumerate(self.test_dataset_loader): 
                description_ts_packed = rnn.pack_sequence(description_ts)
                description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                params_gt_list = [
                    affine_paramss[:,i].view(-1,1) for i in range(affine_paramss.shape[1])
                ]
                prim_pred, normal_dists, pi_dists = self.model(description_ts_packed)  
                # cel = self.ce_loss(prim_pred, primitive_types)
                lls = self.loss(params_gt_list, normal_dists, pi_dists, calculate_mean=False)
                for idx, ll in enumerate(lls):
                    new_sum = ll.sum().item()
                    old_sum = self.test_meter.metric_dict[f'{self.hp.parameter_names[idx]}_loss'] * self.test_meter.count
                    self.test_meter.metric_dict[f'{self.hp.parameter_names[idx]}_loss'] = (new_sum + old_sum) / (len(ll) + self.test_meter.count)
                
                # total_ll = torch.stack(lls).sum()
                # total_lls = total_ll + cel
                # self.test_meter.metric_dict['prim_type_loss'].append(cel.item())
                # self.test_meter.metric_dict['total_param_loss'].append(total_ll.item())
                # for idx, ll in enumerate(lls):
                #     new_loss_sum = ll.sum()
                #     self.test_meter.metric_dict[f'{self.hp.parameter_names[idx]}_loss'] += list(ll)
                # self.test_meter.metric_dict['total_loss'].append(total_lls.item())
                
                prim_types = torch.argmax(prim_pred, dim=1)
                param_samples = self.sample_parameters(normal_dists, pi_dists) # each: N x 1
                np.random.seed(123)
                plot_indices = np.random.choice(len(dataset_indices), self.args.num_visualize, replace=False)
                
                fig, dataset_indices_plot = self.calculate_metric(description_ts, dataset_indices, prim_types, param_samples, plot_indices)  
                image_name = f"{step}-"+",".join([str(x) for x in dataset_indices_plot])
                fig.suptitle(image_name)
                if self.enable_wandb:
                    final_img = plt_to_image(fig)
                    wandb.log({f"evaluate_image_{batch_idx}": wandb.Image(final_img)}, step=step)
                else:
                    image_path = os.path.join(self.save_folder, "{}_{}.png".format(step, batch_idx))
                    plt.savefig(image_path)
                fig.tight_layout()
                plt.close()
            log_dict = self.test_meter.finalize_metric()
            if self.enable_wandb:
                wandb.log(log_dict, step=step)
            print_s = [f"Test @ Iter {step}: "]
            print_dict(log_dict, print_s)
                
    def save_model(self, step):
        torch_path_name = os.path.join(self.save_folder, f"{step}.pt")
        torch.save({
            'iteration' : step,
            'model_state_dict': self.model.state_dict(),
        }, torch_path_name)
    
    def load_model(self, model_path):
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model_state_dict'])

def main():
    wandb.login()
    CONST = Constants()
    args = get_args()
    hp = HParams()
    args_dict = vars(args)
    for k,v in args_dict.items():
        if k == "parameter_names":
            for j,name in enumerate(v):
                hp.parameter_names[j] = name
            continue
        if hasattr(hp, k):
            setattr(hp, k,v)

    vocab = preprocess_dataset_language(args.train_file)
    hp.vocab_size = len(vocab)

    # TEMPLATE_DICT = {
    #     'arc' : lambda n : rd.generate_arc(n1=n, radius=10, x0=0, y0=0, template_size=w),
    #     'circle' : lambda n : rd.generate_circle(n1=n, radius=100, x0=0, y0=0, template_size=w),
    #     'square' : lambda n : rd.generate_square(n1=n, template_size=w),
    #     'semicircle' : lambda n : rd.generate_semicircle(n1=n, radius=100, x0=0, y0=0, template_size=w),
    #     'zigzag1' : lambda n : rd.generate_zigzag1(n1=n, template_size=w),
    # }
    TEMPLATE_DICT = {
        'arc' : lambda n : rd.generate_arc(n1=n, radius=1, x0=0, y0=0, template_size=1),
        'circle' : lambda n : rd.generate_circle(n1=n, radius=1, x0=0, y0=0, template_size=1),
        'square' : lambda n : rd.generate_square(n1=n, template_size=2),
        'semicircle' : lambda n : rd.generate_semicircle(n1=n, radius=1, x0=0, y0=0, template_size=1),
        'zigzag1' : lambda n : rd.generate_zigzag(n1=n, num_fold=1, side_length=1,template_size=1),
    }
    templates = {}
    for i,(k,v) in enumerate(TEMPLATE_DICT.items()):
        arr = v(args.num_sampled_points)
        templates[i] = (k,arr)
    
    trainer = Trainer(hp, args, vocab, templates)
    trainer.train()

if __name__ == "__main__":
    main()