import numpy as np
import pandas as pd
import os
import read_datasets as rd
import constants as CONST
from importlib import reload
reload(rd)
reload(CONST)
import PIL
from PIL import Image
import collections
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import finetune
reload(finetune)
from finetune import *

import argparse 

import wandb
wandb.login()



def get_default_args():
    default_args = {
        'BATCH_SIZE' : 128,
        'EVAL_BATCH_SIZE' : 32,
        'START_EPOCH' : 0,
        'EPOCH' : 50,
        'EVAL_EVERY' : 50,
        'LR' : 5e-6, 
        'BETAS' : (0.9,0.98),
        'EPS' : 1e-6,
        'WEIGHT_DECAY' : 0.01,
        'TRAIN_CATEGORY' : ['face'],#data
        'TEST_CATEGORY' : ['face', 'angel'],
        'IMAGE_PATH_TEMPLATE' : {
                'face' : '/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all/{}.png',
                'angel' : '/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5_all/{}.png',
            },
        'TEMPLATE' : 0, 
        'line_diameter_scale' : [0.25,1.25],
        'default_line_diameter': 10,
        'rotate' : [-1/4*180, 1/4*180],
        'trans' : [0.15, 0.15],
        'scale' : [0.75, 1.25],
        'open_clip' : False,
        'adamw' : False,
        'no_image_augment' : False,
    }


    return default_args

'''
python clip_finetune.py \
--LR 7e-7 \
--EPOCH 300 \
--TRAIN_CATEGORY face \
--adamw \
--no_image_augment \
--GPU 6 \
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--EVAL_BATCH_SIZE",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--START_EPOCH",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train_model_path",
        type=str,
    )
    parser.add_argument(
        "--wandb_id",
        type=int,
    )
    parser.add_argument(
        "--EPOCH",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--EVAL_EVERY",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--LR",
        type=float,
        default=5e-7,
    )
    parser.add_argument(
        "--BETAS",
        nargs='+', 
        type=int,
        default=[0.9,0.98],
    )
    parser.add_argument(
        "--EPS",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--WEIGHT_DECAY",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--TRAIN_CATEGORY",
        nargs='+', 
        type=str,
        default=['face','angel'],
    )
    parser.add_argument(
        "--TEST_CATEGORY",
        nargs='+', 
        type=str,
        default=['face','angel'],
    )
    parser.add_argument(
        "--line_diameter_scale",
        nargs='+', 
        type=float,
        default=[0.25,1.25],
    )
    parser.add_argument(
        "--default_line_diameter",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--rotate",
        nargs='+', 
        type=float,
        default=[-1/4*180, 1/4*180],
    )
    parser.add_argument(
        "--trans",
        nargs='+', 
        type=float,
        default=[0.15, 0.15],
    )
    parser.add_argument(
        "--scale",
        nargs='+', 
        type=float,
        default=[0.75, 1.25],
    )
    parser.add_argument(
        "--FACE_IMAGE_PATH_TEMPLATE",
        type=str,
        default='/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all/{}.png',
    )
    parser.add_argument(
        "--ANGEL_IMAGE_PATH_TEMPLATE",
        type=str,
        default='/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5_all/{}.png',
    )
    parser.add_argument(
        "--TEMPLATE",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--open_clip",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--adamw",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no_image_augment",
        default=False,
        action="store_true",
    )
    


    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test_model_path",
        type=str,
    )
    parser.add_argument(
        "--test_save_path",
        type=str,
    )

    parser.add_argument(
        "--GPU",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


def test(parse_args):
    gpu_num = parse_args['GPU']
    device = "cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu" 
    torch_path_name = parse_args['test_model_path']

    checkpoint = torch.load(torch_path_name)
    args = checkpoint['args']

    use_open_clip = 'open_clip' in args and args['open_clip']
    if use_open_clip:
        import open_clip
    else:
        import clip 

    if use_open_clip:
        model, train_transform, preprocess = \
            open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device = device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        model.load_state_dict(checkpoint['model_state_dict'])

    dfs, train_t_dataloader, test_dataloaders, dev_dataloaders = get_test_loaders(args, get_df(), preprocess)
    final_accs_dup = {
        'test' : {},
        'dev' : {},
    }
    preds = {
        'test' : {},
        'dev' : {},
    }
    gts = {
        'test' : {},
        'dev' : {},
    }

    pred, gt = evaluate(model, train_t_dataloader, device, use_open_clip=use_open_clip)
    acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
    print("Final TRAIN: acc={:.3f}".format(acc))

    preds['train'] = pred
    gts['train'] = gt
    final_accs_dup['train'] = acc

    for cat in args['TEST_CATEGORY']:
        pred, gt = evaluate(model, test_dataloaders[cat], device, use_open_clip=use_open_clip)
        acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
        print("Final TEST: {} acc={:.3f}".format(cat, acc))

        final_accs_dup['test'][cat] = acc
        preds['test'][cat] = pred
        gts['test'][cat] = gt
    
    for cat in args['TEST_CATEGORY']:
        pred, gt = evaluate(model, dev_dataloaders[cat], device, use_open_clip=use_open_clip)
        acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
        print("Final DEV: {} acc={:.3f}".format(cat, acc))

        final_accs_dup['dev'][cat] = acc
        preds['dev'][cat] = pred
        gts['dev'][cat] = gt
    
    acc_path_name = parse_args['test_save_path']
    print("Saving test results to: ", acc_path_name)
    import pickle
    with open(acc_path_name, 'wb+') as f:
        pickle.dump((args['TRAIN_CATEGORY'], final_accs_dup, preds, gts), f)


def main(parse_args):
    args = get_default_args()

    for k,v in args.items():
        if k == 'IMAGE_PATH_TEMPLATE':
            args['IMAGE_PATH_TEMPLATE']['face'] = parse_args['FACE_IMAGE_PATH_TEMPLATE']
            args['IMAGE_PATH_TEMPLATE']['angel'] = parse_args['ANGEL_IMAGE_PATH_TEMPLATE']
        # if k == "FACE_IMAGE_PATH_TEMPLATE":
        #     default_args['IMAGE_PATH_TEMPLATE']['face'] = v
        # elif k == "ANGEL_IMAGE_PATH_TEMPLATE":
        #     default_args['IMAGE_PATH_TEMPLATE']['angel'] = v
        elif k == "BETAS":
            args['BETAS'] = (parse_args['BETAS'][0], parse_args['BETAS'][1])
        else:
            if k not in parse_args:
                print("WARNING: {} not in parse args".format(k))
            else:
                args[k] = parse_args[k] 
    
    
    gpu_num = parse_args['GPU']
    print("Running on GPU: ", gpu_num)
    use_open_clip = 'open_clip' in args and args['open_clip']
    if use_open_clip:
        print("Using OpenCLIP")
        import open_clip
    else:
        import clip 
    
    if parse_args['wandb_id'] is None or parse_args['wandb_id'] == "":
        print("Generating new wandb run.")
        run_id = wandb.util.generate_id()
    else:
        run_id = parse_args['wandb_id']

    # run = wandb.init(project="clip-finetune", config=args)
    run = wandb.init(project="clip-finetune", id=run_id, resume="allow", config=args)
    print("Wandb run name: ", wandb.run.name)
    wandb.config.update(args)

    device = "cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    if use_open_clip:
        model, _, preprocess = \
            open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device = device)
    else:
        model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        if device == "cpu":
            model.float()
        else :
            clip.model.convert_weights(model) 
    
    if parse_args['wandb_id'] is not None and parse_args['wandb_id'] != "":
        torch_path_name = parse_args['train_model_path']
        checkpoint = torch.load(torch_path_name)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    dfp = get_df()
    dfs, train_t_dataloader, test_dataloaders, dev_dataloaders = get_test_loaders(args, dfp, preprocess)
    train_dataloader = get_train_loader(args, dfs)
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if args['adamw']:
        optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args['WEIGHT_DECAY']},
                ],
                lr=args['LR'],
                betas=args['BETAS'],
                eps=args['EPS'],
            )
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr = args['LR'],
            betas = args['BETAS'],
            eps = args['EPS'],
            weight_decay = args['WEIGHT_DECAY'],
        )

    accs = collections.defaultdict(list)
    step = 0
    root_folder = '/raid/xiaoyuz1/clip_model_checkpoint'
    torch_path_name = os.path.join(root_folder, "{}.pt".format(wandb.run.name))
    acc_path_name = os.path.join(root_folder, "{}.pickle".format(wandb.run.name))
    print(torch_path_name, acc_path_name)

    highest_acc = {
        'face' : -1,
        'angel' : -1,
    }
    save_model = False
    model.train()

    for epoch in range(args['START_EPOCH'], args['START_EPOCH']+args['EPOCH']):
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, list_txt = batch
            images = images.to(device)
            if use_open_clip:
                texts = open_clip.tokenize(list_txt).to(device)
                
                image_features, text_features, logit_scale = model(images, texts)
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logit_scale * text_features @ image_features.T
                
            else:
                texts = clip.tokenize(list_txt).to(device)
                logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(min(args['BATCH_SIZE'], len(texts)), dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            wandb.log({"loss":total_loss.item()}, step=step)
            
            
            if use_open_clip:
                optimizer.step()
            else:
                if device == "cpu":
                    optimizer.step()
                else : 
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

            if step % args['EVAL_EVERY'] == 0:
                wandb_dict = {}
                
                pred, gt = evaluate(model, train_t_dataloader, device, use_open_clip=use_open_clip)
                acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
                wandb_dict['train_acc'] = acc
                
                for cat in args['TEST_CATEGORY']:
                    pred, gt = evaluate(model, dev_dataloaders[cat], device, use_open_clip=use_open_clip)
                    acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
                    if len(accs[cat]) == 0 or acc >= highest_acc[cat]:
                        highest_acc[cat] = acc
                        save_model = True
                    
                    accs[cat].append(acc)
                    wandb_dict['dev_{}_acc'.format(cat)] = acc
                
                wandb.log(wandb_dict, step=step)
                
                if save_model:
                    wandb_dict2 = {}
                    for cat in args['TEST_CATEGORY']:
                        pred, gt = evaluate(model, test_dataloaders[cat], device, use_open_clip=use_open_clip)
                        acc = accuracy_score(np.argmax(pred, axis=1).reshape(-1,), gt)
                        wandb_dict2['test_{}_acc'.format(cat)] = acc
                    wandb.log(wandb_dict2, step=step)
                    
                    torch.save(
                    {
                        'epoch': epoch,
                        'iteration' : step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss,
                        'args': args,
                    }, torch_path_name)
                    wandb.save(torch_path_name)
                save_model = False
            
            step += 1
    
    

if __name__ == "__main__":
    parse_args = parse_args()
    if parse_args['test']:
        test(parse_args)
    else:
        main(parse_args)