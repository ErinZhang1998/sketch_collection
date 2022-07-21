from cmath import e, isclose
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
import torchvision.transforms as transforms

import json
from sklearn.neighbors import NearestNeighbors
from primitive_model import *

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
        self.angel_parts_idx_dict_doodler = {
            0:"halo",1 : "angel eyes",2:"angel nose",3:"angel mouth",4:"angel face",5:"body", 6: "limbs", 7:"wings",
        }
CONST = Constants()

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

def rescale_stroke(stroke, canvas_size, original_size):
    stroke_new = np.copy(stroke)
    stroke_new[:, 0] *= canvas_size/original_size
    stroke_new[:, 1] *= canvas_size/original_size
    return stroke_new

def image_bit(img, bkg = 255):
    img[np.isclose(img, bkg, atol=1e-05)] = 0
    img[not np.isclose(img, bkg, atol=1e-05)] = 1
    return img

def prepare_data(df, templates, img_root_path, save_path_prefix = None, line_diameter=3, canvas_size = 64, num_sampled_points = 200, use_projective = False):
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
    
    face_d = rd.get_previous_and_current_strokes(face_sketches)
    angel_d = rd.get_previous_and_current_strokes(angel_sketches)

    for i in tqdm(range(len(df))):
        entry = df.iloc[i]
        if entry['category'] == 'face':
            _,_,current_strokes, previous_strokes = face_d[entry['image_1']][entry['part']]
        else:
            _,_,current_strokes, previous_strokes = angel_d[entry['image_1']][entry['part']]
        current_strokes = [rescale_stroke(x, canvas_size, 256) for x in current_strokes]
        previous_strokes = [rescale_stroke(x, canvas_size, 256)[:,:2] for x in previous_strokes]
        data = rd.process_quickdraw_to_part_convex_hull(current_strokes, b_spline_num_sampled_points=num_sampled_points)
        img_path = os.path.join(img_root_path, f"{entry['category']}_{entry['image_1']}_{entry['part']}_history.png")

        img = rd.render_img(previous_strokes, img_path=img_path, side=canvas_size, line_diameter=line_diameter, original_side = canvas_size)
        
        min_template_squared_error = np.inf
        min_M = None
        min_template_idx = None
        min_result = None
        for template_idx, (_, template) in templates.items():
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
                min_result = result
        transform_mat = np.asarray(min_M).astype(float).reshape(3,3)
        theta, scale_mat, shear_mat = rd.decompose_affine(transform_mat)

        a,b,c,d,e,f,_,_,_ = min_M
        info = {
            'category' : entry['category'],
            'image_idx' : entry['image_1'],
            'part' : entry['part'],
            'raw' : entry['text_1'],
            'processed' : entry['no_punc_str_1'],
            'primitive_type' : int(min_template_idx),
            'image_path' : img_path,
            'M' : min_M,
            "p0" : a,
            "p1" : b,
            "p2" : c,
            "p3" : d,
            "p4" : e,
            "p5" : f,
            "theta" : theta,
            "sx" : scale_mat[0][0],
            "sy" : scale_mat[1][1],
            "hx" : shear_mat[0][1],
            "tx" : transform_mat[0][2],
            "ty" : transform_mat[1][2],
            'error' : min_template_squared_error,
            'split' : split_dict[i],
            'data' : data,
            'result' : min_result,
            "current_strokes" : current_strokes,
            "previous_strokes" : previous_strokes,
        }
        all_data.append(info)
    if save_path_prefix is not None:
        assert ".pkl" not in save_path_prefix
        with open(f"{save_path_prefix}.pkl", "wb") as f:
            save_dict = {
                "all" : all_data,
                "line_diameter" : line_diameter,
                "canvas_size" : canvas_size,
                "num_sampled_points" : num_sampled_points,
                "use_projective" : use_projective,
            }
            pickle.dump(save_dict, f)

        for t in ["train", "dev", "test"]:
            data_split = [x for x in all_data if x["split"] == t]
            # data_split_dict = dict(zip(range(len(data_split)), data_split))
            save_dict = {
                "all" : data_split,
                "line_diameter" : line_diameter,
                "canvas_size" : canvas_size,
                "num_sampled_points" : num_sampled_points,
                "use_projective" : use_projective,
            }
            with open(f"{save_path_prefix}_{t}.pkl", "wb+") as f:
                pickle.dump(data_split, f)

    return all_data

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

class HParams():
    def __init__(self):
        self.word_embed_dim = 64
        self.lstm_output_dim = 512
        self.lstm_layers = 2 
        self.lstm_drop_prob = 0.4
        self.num_primitives = 5
        self.parameter_names = {0:"theta", 1:"sx", 2:"sy", 3:"hx", 4:"tx", 5:"ty"}
        self.get_affine_func_name = "get_affine_transformation"
        self.vocab_size = None
        self.M = 3
        self.weight_decay = 0.0
        self.batch_size = 64
        self.lr = 0.001
        # image_encoder and co-attention
        self.canvas_size = 64
        self.input_channel = 1 
        self.cnn_encoder_filters = [16, 32, 64, 128, 512]
        self.coatt_hidden_dim = 512 
        self.combined_dim = 30

'''
python primitive_selector.py \
    -evaluate \
    -old_vocab_func \
    -M 10 \
    -saved_model_path /raid/xiaoyuz1/doodler_model_checkpoint/legendary-sun-11/55001.pt \
    -vocab_file "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train.pkl"

CUDA_VISIBLE_DEVICES=3 python primitive_selector.py -num_epochs 400 -vocab_file /raid/xiaoyuz1/primitive_selector_training_data/july_18_train.pkl -enable_wandb -parameter_names p0 p1 p2 p3 p4 p5 -get_affine_func_name get_affine_transformation_2 -M 5 -lr 0.001
'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-evaluate", action='store_true')
    parser.add_argument("-saved_model_path", type=str)
    parser.add_argument("-old_vocab_func", action='store_true')
    parser.add_argument("-vocab_file", type=str)
    parser.add_argument("-enable_wandb", action='store_true')
    parser.add_argument("-use_image", action='store_true')
    parser.add_argument("-start_epoch", type=int, default=0)
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-wandb_project_name", type=str, default="doodler-draw")
    parser.add_argument("-wandb_project_entity", type=str, default="erinz")
    parser.add_argument("-save_root_folder", type=str, default="/raid/xiaoyuz1/doodler_model_checkpoint")
    parser.add_argument("-train_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_18_train.pkl")
    parser.add_argument("-dev_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_18_dev.pkl")
    parser.add_argument("-test_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_18_test.pkl")
    parser.add_argument("-train_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_train_sequences.pkl")
    parser.add_argument("-dev_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_dev_sequences.pkl")
    parser.add_argument("-test_seq_file", type=str, default = "/raid/xiaoyuz1/primitive_selector_training_data/july_15_test_sequences.pkl")
    
    parser.add_argument("-word_embed_dim", type=int, default=64)
    parser.add_argument("-lstm_output_dim", type=int, default=512)
    parser.add_argument("-lstm_layers", type=int, default=2)
    parser.add_argument("-lstm_drop_prob", type=float, default=0.2)
    parser.add_argument("-num_primitives", type=int, default=5)
    parser.add_argument("-parameter_names", nargs='+', default=["theta","sx","sy","hx","tx","ty"])
    parser.add_argument("-get_affine_func_name", type=str, default="get_affine_transformation")
    parser.add_argument("-vocab_size", type=int)
    parser.add_argument("-M", type=int, default=2)
    parser.add_argument("-weight_decay", type=float, default=0.0)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.0001)

    parser.add_argument("-canvas_size", type=int, default=256)
    parser.add_argument("-input_channel", type=int, default=1)
    parser.add_argument("-cnn_encoder_filters", nargs='+', default=[16, 32, 64, 128, 512])
    parser.add_argument("-combined_dim", type=int, default=512)
    parser.add_argument("-coatt_hidden_dim", type=int, default=30)

    parser.add_argument("-save_every", type=int, default=1000)
    parser.add_argument("-print_every", type=int, default=500)
    parser.add_argument("-num_visualize", type=int, default=9)
    parser.add_argument("-num_sampled_points", type=int, default=200)
    parser.add_argument("-use_projective", action="store_true")
    

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

def get_affine_transformation_2(info):
    T = np.asarray([
        [info["p0"],info["p1"],info["p2"]],
        [info["p3"],info["p4"],info["p5"]],
        [0 ,0 ,1 ]]
    )
    return T

GET_AFFINE_FUNCS = {
    "get_affine_transformation" : lambda info : get_affine_transformation(info),
    "get_affine_transformation_2" : lambda info : get_affine_transformation_2(info),
}

def preprocess_dataset_language(path, old=False):
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
    if not old:
        for k,v in CONST.face_parts_idx_dict_doodler.items():
            [q2i[x] for x in v.lower().strip().split(" ")]
        for k,v in CONST.angel_parts_idx_dict_doodler.items():
            [q2i[x] for x in v.lower().strip().split(" ")]
    return q2i



def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def collate_primitivedataset(seq_list):
    lens = [len(x["description"]) for x in seq_list]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    output_dict = {}
    output_dict["description"] = [seq_list[i]["description"] for i in seq_order]
    for k in seq_list[0].keys():
        if k == "description":
            continue
        output_dict[k] = torch.stack([seq_list[i][k] for i in seq_order])
    # import pdb; pdb.set_trace()
    return output_dict
    
    description_ts, primitive_types, affine_paramss, indices = zip(*seq_list)
    lens = [len(x) for x in description_ts]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    description_ts = [description_ts[i] for i in seq_order]
    
    primitive_types = torch.stack([primitive_types[i] for i in seq_order])
    affine_paramss = torch.stack([affine_paramss[i] for i in seq_order])
    indices = torch.stack([indices[i] for i in seq_order])
    
    return description_ts, primitive_types, affine_paramss, indices


class PrimitiveDataset(Dataset):
    def __init__(self, path, vocab, parameter_names, use_image = False, img_transform = None):
        super().__init__()
        self.path = path        
        self.data_raw = pickle.load(open(self.path, "rb"))
        # self.df = pd.DataFrame(self.data_raw)

        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.vocab_reverse = dict(zip(vocab.values(), vocab.keys()))
        self.parameter_names = parameter_names
        self.use_image = use_image
        self.img_transform = img_transform if img_transform is not None else transforms.Compose([transforms.ToTensor()]) 

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, index):
        # Process language input 
        info = self.data_raw[index]
        if info['category'] == 'face':
            description = "{} {}".format(info['processed'], CONST.face_parts_idx_dict_doodler[info['part']])
        else:
            description = "{} {}".format(info['processed'], CONST.angel_parts_idx_dict_doodler[info['part']])
        
        description_t = [self.vocab[x.lower()] for x in description.split(" ") if x.lower() in self.vocab_keys]
        description_t = torch.from_numpy(np.array(description_t)).long()

        primitive_type = torch.tensor(info['primitive_type']).long()
        param_list = [info[k] for _,k in self.parameter_names.items()]
        affine_params = torch.FloatTensor(np.array(param_list))
        
        data_dict = {
            "description" : description_t,
            "primitive_type" : primitive_type,
            "affine_params" : affine_params,
            "index" : torch.tensor(index).long()
        }
        if self.use_image:
            img = self.transform(PIL.Image.open(info["image_path"]))
            data_dict["image"] = img

        return data_dict

def chamfer_distance(x, y):
  x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
  min_y_to_x = x_nn.kneighbors(y)[0]
  y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
  min_x_to_y = y_nn.kneighbors(x)[0]
  dist_y_to_x = np.mean(min_y_to_x)
  dist_x_to_y = np.mean(min_x_to_y)
  return dist_y_to_x + dist_x_to_y#, dist_y_to_x, dist_x_to_y

def mse_metric(gt_img,pred_img):
    return np.mean((gt_img.astype(np.float32) - pred_img.astype(np.float32)) ** 2)

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
    
    def log_metric(self, pred_type, gt_type, calculated_metrics):
        if pred_type == gt_type:
            self.correct_count += 1 
        self.count += 1
        for k,v in calculated_metrics.items():
            new_v = self.metric_dict[k] + (v - self.metric_dict[k]) / self.count
            self.metric_dict[k] = new_v 
    
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


def visualize_fitted_primitive(ax, plot_data_dict, last_ax = False):
    ax.scatter(plot_data_dict["original_data"][:,0], plot_data_dict["original_data"][:,1], s=1, c='b', label = "original data")
    ax.scatter(plot_data_dict["gt_template_gt_param"][:,0], plot_data_dict["gt_template_gt_param"][:,1], s=1, alpha=0.5, c='r', label="GT Fitted Primitive")
    ax.scatter(plot_data_dict["gt_template_pred_param"][:,0], plot_data_dict["gt_template_pred_param"][:,1], s=1, c='darkgreen', alpha=0.5, label="GT Primitive Pred Params")
    ax.scatter(plot_data_dict["pred_template_pred_param"][:,0], plot_data_dict["pred_template_pred_param"][:,1], s=1, c='lime', label="Pred Primitive Pred Params")
    ax.axis(xmin=-plot_data_dict["canvas_size"],xmax=plot_data_dict["canvas_size"])
    ax.axis(ymin=plot_data_dict["canvas_size"],ymax=-plot_data_dict["canvas_size"])
    
    gt_template_name = plot_data_dict["gt_template_name"]
    pred_template_name = plot_data_dict["pred_template_name"]
    desc = plot_data_dict["original_description"]
    desc2 = plot_data_dict["processed_description"]
    
    mse = plot_data_dict["mse"]
    psnr = plot_data_dict["psnr"]
    ssim = plot_data_dict["ssim"]
    chamfer_dist = plot_data_dict["chamfer_dist"]
    
    title = f"{gt_template_name},{pred_template_name}\n" 
    title += f"{desc}\n{desc2}\n"
    title += f"{mse:.2f} {psnr:.2f} {ssim:.2f} {chamfer_dist:.2f}\n"
    
    parameter_names = plot_data_dict["parameter_names"]
    for pi,(p1,p2) in enumerate(zip(plot_data_dict["gt_params"], plot_data_dict["pred_params"])):
        if parameter_names[pi] == "theta":
            p1 = np.degrees(p1)
            p2 = np.degrees(p2)
        title += f"{parameter_names[pi]}: {p1:.3f} {p2:.3f}"
        if pi % 2 == 0:
            title += "\n"
        else:
            title += " | "
    ax.set_title(title, y=0.7)
    # ax.legend()
    if last_ax:
        handles, labels = ax.get_legend_handles_labels()
        return handles, labels 
    else:
        return None, None    

class Trainer():
    def __init__(self, hp, args, vocab, templates, img_encoder = None):
        
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
        
        self.train_dataset = PrimitiveDataset(args.train_file, vocab, hp.parameter_names, use_image=args.use_image)
        self.test_dataset = PrimitiveDataset(args.test_file, vocab, hp.parameter_names, use_image=args.use_image)
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
        self.model = PrimitiveSelector(hp, img_enc=img_encoder).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.test_meter = Meter("test")
        self.train_meter = Meter("train")
        
        self.num_pngs_per_row = 3
    
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
    
    def calculate_metric(self, descriptions, dataset_indices, prim_types, param_samples, plot_indices=None, train=False, plot = False):
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
        
        dataset_indices_plot = []
        plot_indices = plot_indices if plot_indices is not None else np.arange(len(dataset_indices))
        if plot:
            num_visualize = len(plot_indices)
            num_rows = num_visualize // self.num_pngs_per_row
            if num_rows * self.num_pngs_per_row < num_visualize:
                num_rows += 1
            # fig = plt.figure(figsize=(self.num_pngs_per_row * 5, num_rows * 5)) 
            fig, axes = plt.subplots(num_rows, self.num_pngs_per_row, figsize=(self.num_pngs_per_row * 5, num_rows * 5))
            fig.patch.set_alpha(1)  

        plot_i = 0
        for idx in range(len(dataset_indices)):
            description_tensor = descriptions[idx]
            data_idx = dataset_indices[idx].item()
            prim_type = prim_types[idx].item()
            pred_params = [sample[idx].item() for sample in param_samples]
            if train:
                info = self.train_dataset.data_raw[data_idx]
            else:
                info = self.test_dataset.data_raw[data_idx]
           
            pred_template_name, pred_template = self.templates[int(prim_type)]
            gt_template_name,_ = self.templates[int(info["primitive_type"])]
            if train:
                data, gt_template, gt_fitted_template, gt_img = self.train_sequences[data_idx]
            else:
                data, gt_template, gt_fitted_template, gt_img = self.test_sequences[data_idx]
            gt_params = [info[self.hp.parameter_names[pi]] for pi in range(len(self.hp.parameter_names))]
            
            pred_info = {}
            for pi in range(len(self.hp.parameter_names)):
                pred_info[self.hp.parameter_names[pi]] = pred_params[pi]
            pred_M = GET_AFFINE_FUNCS[self.hp.get_affine_func_name](pred_info)
            
            if self.args.use_projective:
                pred_template_pred_param = cv2.perspectiveTransform(pred_template, pred_M).reshape(-1,2)
            else:
                pred_template_pred_param = cv2.transform(np.array([pred_template]).astype(np.float32), pred_M)[0][:,:-1]

            # gt_img = np.asarray(rd.render_img([gt_fitted_template], line_diameter=3))
            pred_img = np.asarray(rd.render_img([pred_template_pred_param], line_diameter=3))
            chamfer_dist = chamfer_distance(data, pred_template_pred_param)
            mse = mse_metric(gt_img, pred_img)
            psnr = 100 if np.isclose(mse, 0.0, rtol=0.0, atol=1e-5) else 20 * np.log10(255 / (np.sqrt(mse)))
            ssim = structural_similarity(gt_img, pred_img, multichannel=False, data_range=255)
            calculated_metrics = {
                "mse" : mse,
                "psnr" : psnr,
                "ssim" : ssim,
                "chamfer" : chamfer_dist,
            }
            if train:
                self.train_meter.log_metric(prim_type, int(info["primitive_type"]), calculated_metrics)
            else:
                self.test_meter.log_metric(prim_type, int(info["primitive_type"]), calculated_metrics)
            
            if plot and idx in plot_indices:
                ax = axes[plot_i // self.num_pngs_per_row][plot_i % self.num_pngs_per_row]
                dataset_indices_plot.append(data_idx)
                if self.args.use_projective:
                    correct_template_pred_param = cv2.perspectiveTransform(gt_template, pred_M).reshape(-1,2)
                else:
                    correct_template_pred_param = cv2.transform(np.array([gt_template]).astype(np.float32), pred_M)[0][:,:-1]
                if self.args.use_projective:
                    pred_template_pred_param = cv2.perspectiveTransform(pred_template, pred_M).reshape(-1,2)
                else:
                    pred_template_pred_param = cv2.transform(np.array([pred_template]).astype(np.float32), pred_M)[0][:,:-1]
                desc2 = " ".join([self.train_dataset.vocab_reverse[j.item()] for j in description_tensor])
                plot_data_dict = {
                    "original_data" : data,
                    "gt_template_gt_param" : gt_fitted_template,
                    "gt_template_pred_param": correct_template_pred_param,
                    "pred_template_pred_param": pred_template_pred_param,
                    "canvas_size" : self.args.canvas_size,
                    "gt_template_name" : gt_template_name,
                    "pred_template_name" : pred_template_name,
                    "original_description" : info["processed"],
                    "processed_description" : desc2,
                    "mse" : mse,
                    "psnr" : psnr,
                    "ssim" : ssim,
                    "chamfer_dist" : chamfer_dist,
                    "parameter_names" : self.hp.parameter_names,
                    "gt_params" : gt_params,
                    "pred_params" : pred_params,
                }
                handles, labels = visualize_fitted_primitive(ax, plot_data_dict, last_ax=plot_i == num_visualize-1)
                
                plot_i += 1
        
        if plot:
            fig.legend(handles, labels, loc='lower left')
            return fig, dataset_indices_plot
       
    def train(self):
        self.model.train()
        step = 0
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.num_epochs):
        
            for batch_idx, output_dict in enumerate(self.train_dataset_loader):
                description_ts = output_dict["description"]
                primitive_types = output_dict["primitive_type"]
                affine_paramss = output_dict["affine_params"]
                dataset_indices = output_dict["index"]
                description_ts_packed = rnn.pack_sequence(description_ts)
                
                description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                params_gt_list = [affine_paramss[:,i].view(-1,1) for i in range(affine_paramss.shape[1])]

                # prim_pred, pi_list, mu_list, sigma_list = self.model(description_ts_packed)
                if self.args.use_image:
                    images = output_dict["image"].to(self.device)
                else: 
                    images = None
                prim_pred, normal_dists, pi_dists = self.model(description_ts_packed, image=images)         
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
                        for batch_idx, output_dict in enumerate(self.train_dataset_loader):
                            description_ts = output_dict["description"]
                            primitive_types = output_dict["primitive_type"]
                            affine_paramss = output_dict["affine_params"]
                            dataset_indices = output_dict["index"]
                            description_ts_packed = rnn.pack_sequence(description_ts)
                            description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                            prim_pred, normal_dists, pi_dists = self.model(description_ts_packed)
                            prim_types = torch.argmax(prim_pred, dim=1)
                            param_samples = self.sample_parameters(normal_dists, pi_dists) # each: N x 1   
                            _ = self.calculate_metric(description_ts, dataset_indices, prim_types, param_samples, plot_indices=None, train=True, plot=False)
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
            for batch_idx, output_dict in enumerate(self.test_dataset_loader):
                description_ts = output_dict["description"]
                primitive_types = output_dict["primitive_type"]
                affine_paramss = output_dict["affine_params"]
                dataset_indices = output_dict["index"]
                description_ts_packed = rnn.pack_sequence(description_ts)
                description_ts_packed, primitive_types, affine_paramss = description_ts_packed.to(self.device), primitive_types.to(self.device), affine_paramss.to(self.device)
                
                # try: 
                #     description_ts_packed = rnn.pack_sequence(description_ts)
                # except:
                #     print(description_ts)
                #     for ii, descr in enumerate(description_ts):
                #         desc_str = " ".join([self.test_dataset.vocab_reverse[j.item()] for j in descr])
                #         print(batch_idx, dataset_indices[ii].item(), desc_str)
                #     raise 
                
                params_gt_list = [affine_paramss[:,i].view(-1,1) for i in range(affine_paramss.shape[1])]
                images = output_dict["image"].to(self.device) if self.args.use_image else None
                prim_pred, normal_dists, pi_dists = self.model(description_ts_packed, image=images) 
                # cel = self.ce_loss(prim_pred, primitive_types)
                lls = self.loss(params_gt_list, normal_dists, pi_dists, calculate_mean=False)
                for idx, ll in enumerate(lls):
                    new_sum = ll.sum().item()
                    old_sum = self.test_meter.metric_dict[f'{self.hp.parameter_names[idx]}_loss'] * self.test_meter.count
                    # !!!!! IMPORTANT: calculate loss metric before image metric since self.test_meter.count will change!!!
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
                
                fig, dataset_indices_plot = self.calculate_metric(description_ts, dataset_indices, prim_types, param_samples, plot_indices, plot=True)  
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
            'hp' : self.hp.__dict__,
            'iteration' : step,
            'model_state_dict': self.model.state_dict(),
        }, torch_path_name)
    
    def load_model(self, model_path):
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if "hp" in ckpt:
            self.hp = ckpt["hp"]
        return ckpt["iteration"]

def main():
    wandb.login()
    
    args = get_args()
    hp = HParams()
    args_dict = vars(args)
    for k,v in args_dict.items():
        if k == "parameter_names":
            for j,name in enumerate(v):
                hp.parameter_names[j] = name
        elif k == "cnn_encoder_filters":
            hp.image_output_dim = args.cnn_encoder_filters[-1]
        elif hasattr(hp, k):
            setattr(hp, k,v)
        else:
            continue

    lang_file = args.train_file if args.vocab_file is None else args.vocab_file
    vocab = preprocess_dataset_language(lang_file, old=args.old_vocab_func)
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
    
    if args.use_image:
        img_encoder = CNN_Encoder(hp.input_channel, filters = hp.cnn_encoder_filters)
    else:
        img_encoder = None
    trainer = Trainer(hp, args, vocab, templates, img_encoder)
    if args.evaluate:
        step = trainer.load_model(args.saved_model_path)
        trainer.evaluate(step)
    else:
        trainer.train()

if __name__ == "__main__":
    main()