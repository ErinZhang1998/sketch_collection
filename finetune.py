import constants as CONST
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import read_datasets as rd
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pandas as pd
import pickle

def template_type1_1(x):
    if x['category'] == 'face':
        return "{} {}".format(x['no_punc_str_1'], CONST.face_parts_idx_dict[x['part']])
    else:
        return "{} {}".format(x['no_punc_str_1'], CONST.angel_parts_idx_dict[x['part']])

def template_type1_2(x):
    if x['category'] == 'face':
        return "{} {}".format(x['no_punc_str_2'], CONST.face_parts_idx_dict[x['part']])
    else:
        return "{} {}".format(x['no_punc_str_2'], CONST.angel_parts_idx_dict[x['part']])

def template_type2_1(x):
    if x['category'] == 'face':
        return "{} {}".format(x['text_1'], CONST.face_parts_idx_dict[x['part']])
    else:
        return "{} {}".format(x['text_1'], CONST.angel_parts_idx_dict[x['part']])

def template_type2_2(x):
    if x['category'] == 'face':
        return "{} {}".format(x['text_2'], CONST.face_parts_idx_dict[x['part']])
    else:
        return "{} {}".format(x['text_2'], CONST.angel_parts_idx_dict[x['part']])


template = {
    0 : (lambda x: template_type1_1(x), lambda x: template_type1_2(x)),
    1 : (lambda x: template_type2_1(x), lambda x: template_type2_2(x))
}

class ImageTextAsOneDataset(Dataset):
    def __init__(self, df, img_path_template, preprocess):
        self.preprocess = preprocess
        self.img_path_template = img_path_template
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        img_path = self.img_path_template.format(self.images[idx])
        images = self.preprocess(Image.open(img_path)) 
        caption = self.caption[idx]
        return images, caption

class ImageTextDataset(Dataset):
    def __init__(self, df, img_path_template, preprocess, text_preprocess, line_diameter_scale=[0.25,1.25],default_line_diameter=10):
        self.preprocess = preprocess
        self.text_preprocess = text_preprocess
        self.images = df["image_1"].tolist() + df["image_2"].tolist()
        self.caption = df["caption_1"].tolist() + df["caption_2"].tolist()
        self.img_path_template = img_path_template
        self.categories = df['category'].tolist() + df['category'].tolist()     

        self.image_to_vector = {'face' : {}, 'angel' : {}}
        for idx in CONST.all_face_indices:
            vector_part = rd.create_im(CONST.face_json, idx, part_idxs=[])
            self.image_to_vector['face'][idx] = vector_part
        for idx in CONST.all_angel_indices:
            vector_part = rd.create_im(CONST.angel_json, idx, part_idxs=[])
            self.image_to_vector['angel'][idx] = vector_part
        
        self.line_diameter_scale = line_diameter_scale
        self.default_line_diameter = default_line_diameter

    def __len__(self):
        return len(self.caption) 

    def __getitem__(self, idx):
        vector_part = self.image_to_vector[self.categories[idx]][self.images[idx]]
        line_diameter = np.random.uniform(*self.line_diameter_scale)*self.default_line_diameter
        t = rd.render_img(vector_part, line_diameter=line_diameter, convert=False)
        images = self.preprocess(t) 
        #img_path = self.img_path_template[self.categories[idx]].format(self.images[idx])
        #images = self.preprocess(Image.open(img_path)) 
        caption = self.caption[idx]

        if len(self.text_preprocess) > 0:
            aug = random.choice(self.text_preprocess)
            caption = aug.augment(caption)

        return images, caption

class ImageTextPairDataset(Dataset):
    def __init__(self, df, img_path_template, preprocess):
        self.preprocess = preprocess
        self.images = {
            0 : df["image_1"].tolist(),
            1 : df["image_2"].tolist(),
        }
        self.caption = {
            0 : df["caption_1"].tolist(),
            1 : df["caption_2"].tolist(),
        }
        self.categories = df["category"].tolist()
        self.img_path_template = img_path_template
        
        assert len(self.caption[1]) == len(self.caption[0]) == len(self.images[1]) == len(self.images[0])

    def __len__(self):
        return len(self.caption[0]) 

    def __getitem__(self, idx):
        img_augs = []
        ws = []
        for i in range(2):
            img_path = self.img_path_template[self.categories[idx]].format(self.images[i][idx])
            one_img = self.preprocess(Image.open(img_path)).unsqueeze(0)
            img_augs.append(one_img)
            ws.append(self.caption[i][idx])
        images = torch.cat(img_augs)
        return images, ws

def _convert_to_rgb(image):
    return image.convert('RGB')

def transform(args, n_px = 224, is_train = True, open_clip = False, augment = True):
    normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    if open_clip:
        interpp = T.InterpolationMode.BICUBIC
    else:
        interpp = Image.BICUBIC
    
    if is_train:
        if augment:
            return T.Compose([
                T.RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=interpp),
                T.RandomAffine(degrees=args['rotate'], translate=args['trans'], scale=args['scale']),
                T.RandomHorizontalFlip(p=0.5),
                #T.RandomVerticalFlip(p=0.5),
                _convert_to_rgb,
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=interpp),
                _convert_to_rgb,
                T.ToTensor(),
                normalize,
            ])

    else:
        return T.Compose([
            T.Resize(n_px, interpolation=interpp),
            T.CenterCrop(n_px),
            _convert_to_rgb,
            T.ToTensor(),
            normalize,
        ])

def get_df():
    dfp = pd.read_csv('/raid/xiaoyuz1/amazon_turk/df_all_pair_withsplit.csv')
    dfp['no_punc_1'] = dfp.no_punc_1.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])
    dfp['no_punc_2'] = dfp.no_punc_2.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])
    return dfp

def get_test_loaders(args, dfp, preprocess):
    template_1,template_2 = template[args['TEMPLATE']]
    img_path_template = args['IMAGE_PATH_TEMPLATE']

    dfs = {'test' : {}, 'dev' : {}}
    df = dfp[(dfp['category'].isin(args['TRAIN_CATEGORY'])) & (dfp['split'] == 'train')]
    df = df.reset_index(drop=True)
    df['caption_1'] = df.apply(template_1, axis=1)
    df['caption_2'] = df.apply(template_2, axis=1)
    dfs['train'] = df

    for split in ['test','dev']:
        for cat in args['TEST_CATEGORY']:
            df = dfp[(dfp['category'] == cat) & (dfp['split'] == split)]
            df = df.reset_index(drop=True)
            df['caption_1'] = df.apply(template_1, axis=1)
            df['caption_2'] = df.apply(template_2, axis=1)
            dfs[split][cat] = df
    
    train_t_dataset = ImageTextPairDataset(dfs['train'], img_path_template, preprocess)
    train_t_dataloader = DataLoader(train_t_dataset, batch_size = args['EVAL_BATCH_SIZE'], shuffle=False)
    test_dataloaders = {}
    for cat in args['TEST_CATEGORY']:
        test_dataset = ImageTextPairDataset(dfs['test'][cat], img_path_template, preprocess)
        test_dataloader = DataLoader(test_dataset, batch_size = args['EVAL_BATCH_SIZE'], shuffle=False)
        print("Number of test batches: ", len(test_dataloader))
        
        test_dataloaders[cat] = test_dataloader 
    
    dev_dataloaders = {}
    for cat in args['TEST_CATEGORY']:
        dev_dataset = ImageTextPairDataset(dfs['dev'][cat], img_path_template, preprocess)
        dev_dataloader = DataLoader(dev_dataset, batch_size = args['EVAL_BATCH_SIZE'], shuffle=False)
        print("Number of dev batches: ", len(dev_dataloader))
        
        dev_dataloaders[cat] = dev_dataloader
    
    return dfs, train_t_dataloader, test_dataloaders, dev_dataloaders

def get_train_loader(args, dfs):
    img_path_template = args['IMAGE_PATH_TEMPLATE']
    train_preprocess = transform(args, n_px = 224, is_train = True, open_clip = args['open_clip'], augment = not args['no_image_augment'])
    train_dataset = ImageTextDataset(
        dfs['train'], 
        img_path_template, 
        train_preprocess,
        [],
    )
    train_dataloader = DataLoader(train_dataset, batch_size = args['BATCH_SIZE'], shuffle=True) 
    print("Number of train batches: ", len(train_dataloader))
    return train_dataloader

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def cosine_sim(t1,t2,model,device):
    with torch.no_grad():
        word_feat1 = model.encode_text(torch.cat([clip.tokenize(t1).cuda(device)]))
        word_feat2 = model.encode_text(torch.cat([clip.tokenize(t2).cuda(device)]))
        word_feat1 = word_feat1 / word_feat1.norm(dim=-1, keepdim=True)
        word_feat2 = word_feat2 / word_feat2.norm(dim=-1, keepdim=True)
    return (word_feat1@word_feat2.T).cpu().item()

def calculate_word_similarity_clip(df, model, device, size = 100, column_name="cosine_sim_clip"):
    model.eval()

    word1s = df['word1'].tolist()
    word2s = df['word2'].tolist()

    word_feat1s = []
    word_feat2s = []

    n = len(word1s)

    with torch.no_grad():
        for i in range(0, n, size):
            
            start = i
            end = min(i+ size, n)
            texts1 = word1s[start:end]
            texts2 = word2s[start:end]
            #print(start,end)
            
            word_feat1 = model.encode_text(torch.cat([clip.tokenize(texts1).cuda(device)]))
            word_feat2 = model.encode_text(torch.cat([clip.tokenize(texts2).cuda(device)]))
            word_feat1 = word_feat1 / word_feat1.norm(dim=-1, keepdim=True)
            word_feat2 = word_feat2 / word_feat2.norm(dim=-1, keepdim=True)
            word_feat1s.append(word_feat1.cpu())
            word_feat2s.append(word_feat2.cpu())
    
    cos_sims = []
    word_feat1 = torch.cat(word_feat1s, axis=0)
    word_feat2 = torch.cat(word_feat2s, axis=0)
    for f1,f2 in zip(word_feat1, word_feat2):
        cos_sims.append((f1 @ f2).item())

    df[column_name] = cos_sims
    return df

def calculate_word_similarity_bert(df, model, tokenizer, device, size = 100, column_name="cosine_sim_bert"):
    # model.eval()

    word1s = df['word1'].tolist()
    word2s = df['word2'].tolist()

    word_feat1s = []
    word_feat2s = []

    n = len(word1s)

    with torch.no_grad():
        for i in range(0, n, size):
            
            start = i
            end = min(i+ size, n)
            texts1 = word1s[start:end]
            texts2 = word2s[start:end]
            
            encoded_input1 = tokenizer(texts1, return_tensors='pt',padding=True).to(device)
            encoded_input2 = tokenizer(texts2, return_tensors='pt',padding=True).to(device)
            word_feat1 = model(**encoded_input1).pooler_output
            word_feat2 = model(**encoded_input2).pooler_output
            
            word_feat1 = word_feat1 / word_feat1.norm(dim=-1, keepdim=True)
            word_feat2 = word_feat2 / word_feat2.norm(dim=-1, keepdim=True)
            word_feat1s.append(word_feat1.cpu())
            word_feat2s.append(word_feat2.cpu())
    
    cos_sims = []
    word_feat1 = torch.cat(word_feat1s, axis=0)
    word_feat2 = torch.cat(word_feat2s, axis=0)
    for f1,f2 in zip(word_feat1, word_feat2):
        cos_sims.append((f1 @ f2).item())

    df['cosine_sim_bert'] = cos_sims
    return df

def calculate_word_embeddings(word1s, model, device, tokenizer = None, size = 100, path = None, save_to_path = False, model_type = "clip"):
    word_feat1s = []

    n = len(word1s)
    mask = [True] * n
    if model_type == "bert":
        if tokenizer is None:
            raise ValueError("When using bert model, please provide tokenizer.")
        print("Extracting word embeddings using bert")
    elif model_type == "glove":
        print("Extracting word embeddings using Glove Gensim")
    elif model_type == "clip":
        print("Extracting word embeddings using CLIP")
    else:
        raise ValueError('Please specify a valid model type: bert | glove | clip.')

    acc = 0
    with torch.no_grad():
        for i in range(0, n, size):
            
            start = i
            end = min(i+ size, n)
            texts1 = word1s[start:end]

            if model_type == "bert":
                encoded_input1 = tokenizer(texts1, return_tensors='pt',padding=True).to(device)
                word_feat1 = model(**encoded_input1).pooler_output
            elif model_type == "glove":
                word_feat1_list = []
                for w in texts1:
                    if w not in model:
                        feat = np.asarray([-1] * 300) 
                        mask[acc] = False
                    else:
                        feat = model[w]
                    acc += 1
                    word_feat1_list.append(torch.from_numpy(feat.reshape(1,-1)))
                word_feat1 = torch.cat(word_feat1_list, axis=0).to(device)
            elif model_type == "clip":
                word_feat1 = model.encode_text(torch.cat([clip.tokenize(texts1).cuda(device)]))           
                
            else:
                raise ValueError('Can only process model type = bert | glove | clip.')
                    
            
            word_feat1s.append(word_feat1)
            
                
    word_feat1 = torch.cat(word_feat1s, axis=0)
    # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    if save_to_path and path is not None:
        word_feat1_norm =  word_feat1 / word_feat1.norm(dim=-1, keepdim=True)
        sim = word_feat1_norm @ word_feat1_norm.T
        # sim = cos(word_feat1, word_feat1)
        with open(path, 'wb+') as f:
            pickle.dump((word1s, mask, word_feat1.cpu().numpy(), sim.cpu().numpy()),f)
    return word_feat1

def evaluate(model, loader, device, use_open_clip = False):
    model.eval()
    
    pred = []
    gt = []
    
    with torch.no_grad():
        for batch in loader:
            images, texts = batch
            texts1,texts2 = texts
            
            images_flat = images.view(-1, *images.shape[2:]).contiguous().cuda(device)
            images_feat = model.encode_image(images_flat)
            images_feat = images_feat.view(-1,2,images_feat.shape[-1])
            images_feat = images_feat / images_feat.norm(dim=-1, keepdim=True)
            if use_open_clip:
                import open_clip
                word_feat1 = model.encode_text(torch.cat([open_clip.tokenize(texts1).cuda(device)]))
                word_feat2 = model.encode_text(torch.cat([open_clip.tokenize(texts2).cuda(device)]))
            else:
                word_feat1 = model.encode_text(torch.cat([clip.tokenize(texts1).cuda(device)]))
                word_feat2 = model.encode_text(torch.cat([clip.tokenize(texts2).cuda(device)]))
            word_feat = torch.stack((word_feat1, word_feat2), 1)
            word_feat = word_feat / word_feat.norm(dim=-1, keepdim=True)
            
            for imgi,wordi in zip(images_feat,word_feat):
                prob = imgi @ wordi.T
                prob = list(prob.cpu().numpy())
                pred += prob
                gt += [0,1]
    
    pred = np.asarray(pred).reshape(-1,2,2)
    gt = np.asarray(gt)

    return pred, gt