import numpy as np
import matplotlib.pyplot as plt
import PIL
import pickle
import os 
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torchvision.models as models

from torch.utils.data import DataLoader

from tqdm import tqdm

dataset_mapping = {'butterfly': [[], []],
 'candle': [[0], [1,2]],
 'drill': [[1], [0]],
 'ice_cream': [[1], [0]],
 'pineapple': [[1], [0]],
 'ambulance': [[1], [0]],
 'duck': [[1], [0,3]],
 'basket': [[1], [0]],
 'pig': [[5], [0,1,2,3,4]],
 'suitcase': [[1], [0]],
 'angel': [[5], [0,1,2,3,4]],
 'alarm_clock': [[], []],
 'flower': [[2], [0,1]],
 'house': [[1], [0]],
 'coffee_cup': [[], []],
 'bulldozer': [[], []],
 'calculator': [[], []],
 'face': [[], []],
 'campfire': [[], []],
 'airplane': [[], []],
 'backpack': [[], []],
 'ant': [[3], [0,1,2]],
 'crab': [[2], [0,1]],
 'apple': [[], []],
 'cactus': [[], []]}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HParams():
    def __init__(self):
        self.root_dir = '/raid/xiaoyuz1/sketch_datasets/spg_train'
        self.enc_hidden_size = 8
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 10
        self.dropout = 0.5
        self.batch_size = 32
        self.lr = 0.005
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.max_seq_length = 200
        self.epochs = 20

def preprocess(hp):
    all_files = {}
    for category_name,_ in dataset_mapping.items():
        category_files = []
        category_folder = os.path.join(hp.root_dir, category_name)
        
        if not os.path.exists(category_folder):
            print(category_folder, " not found!")
            continue
        
        for file in os.listdir(category_folder):
            if not file.endswith('.png'):
                continue 
            
            file_root = file.split(".")[0]

            stroke_file = os.path.join(hp.root_dir, category_name, "{}.npz".format(file_root))
            this_data = np.load(stroke_file, encoding='latin1', allow_pickle=True)
            mask = this_data['mask']
            if(np.sum(mask) < 1):
                continue

            category_files.append((category_name, file_root))
        all_files[category_name] = category_files
    return all_files

def get_train_test_files(hp):
    all_files = preprocess(hp)
    # acc = 0
    # for k,v in all_files.items():
    #     acc += len(v)
    #     print(k, len(v))
    # print(acc)

    train_files = []
    test_files = []

    for k,v in all_files.items():
        batch_idx = np.random.choice(len(v), 50)
        test_files += [v[idx] for idx in batch_idx]
        for idx in np.arange(len(v)):
            if idx in batch_idx:
                continue 
            train_files.append(v[idx])
    
    return train_files,test_files

def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class StartDataset(Dataset):
    def __init__(self, root_dir, train_files):
        self.root_dir = root_dir

        self.image_transforms = transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor()
        ])

        self.all_corners, self.train_files = self.pre_load_strokes(train_files)
    
    def pre_load_strokes(self, train_files):
        all_corners = {}
        train_files_new = []
        for category_name,file_root in train_files:
            k = "{}-{}".format(category_name, file_root)
            stroke_file = os.path.join(self.root_dir, category_name, "{}.npz".format(file_root))
            this_data = np.load(stroke_file, encoding='latin1', allow_pickle=True)
            absolute_coord = this_data['absolute_stroke']
            mask = this_data['mask']
            if np.sum(mask) < 1:
                print("WRANING! ", k)
                continue
            arr = np.asarray(absolute_coord)[mask]
            x1,y1 = arr[:,0].min(), arr[:,1].min()
            x2,y2 = arr[:,0].max(), arr[:,1].max()
            x,y = (x1+x2)*0.5, (y1+y2)*0.5
            all_corners[k] = [x/256.0, y/256.0]
            train_files_new.append((category_name,file_root))

        return all_corners, train_files_new
    
    def __len__(self):
        return len(self.train_files)
    
    def __getitem__(self, idx):
        category_name, file_root = self.train_files[idx]
        img_file = os.path.join(self.root_dir, category_name, "{}.png".format(file_root))
        k = "{}-{}".format(category_name, file_root)
        x,y = self.all_corners[k]

        img = PIL.Image.open(img_file)
        img = self.image_transforms(img) 
        return img[:1,:,:], torch.Tensor([x,y]).float()

class PartDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass 


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet18(pretrained=False)
        # modules = list(resnet.children())[:-2]
        # self.resnet = nn.Sequential(*modules)
        # # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # # self.fine_tune()

        '''
        x = self.conv_2d('conv1', x, filter_size=2, out_filters=4, strides=self.stride_arr(2))  # [N, 24, 24, 4]
        x = tf.nn.relu(x)
        x = self.conv_2d('conv2', x, filter_size=2, out_filters=4, strides=self.stride_arr(1))  # [N, 24, 24, 4]
        x = tf.nn.relu(x)
        x = self.conv_2d('conv3', x, filter_size=2, out_filters=8, strides=self.stride_arr(2))  # [N, 12, 12, 8]
        x = tf.nn.relu(x)
        x = self.conv_2d('conv4', x, filter_size=2, out_filters=8, strides=self.stride_arr(1))  # [N, 12, 12, 8]
        x = tf.nn.relu(x)
        x = self.conv_2d('conv5', x, filter_size=2, out_filters=8, strides=self.stride_arr(2))  # [N, 6, 6, 8]
        x = tf.nn.relu(x)
        x = self.conv_2d('conv6', x, filter_size=2, out_filters=8, strides=self.stride_arr(1))  # [N, 6, 6, 8]
        x = tf.tanh(x)

        '''

        modules = []
        hidden_dims = [4, 8, 8]
        in_channels = 1 # 
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=2, stride= 2, padding=0),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            modules.append(
                nn.Sequential(
                    nn.Conv2d(h_dim, out_channels=h_dim, kernel_size=2, stride=1, padding=0),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)



    def forward(self, images):
        out = self.encoder(images)
        # out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1) #(B,H,W,C)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for c in list(self.encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DecoderRNN(nn.Module):
    def __init__(self, dec_hidden_size, Nz, M, dropout):
        super(DecoderRNN, self).__init__()
        self.dec_hidden_size = dec_hidden_size
        self.Nz = Nz
        self.M = M
        self.fc_hc = nn.Linear(Nz, 2 * dec_hidden_size)
        self.lstm = nn.LSTM(Nz+5, dec_hidden_size, dropout=dropout)
        self.fc_params = nn.Linear(dec_hidden_size, 6 * M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(F.tanh(self.fc_hc(z)),self.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        outputs,(hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, self.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params: (2,N,1)
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        # preprocess params::
        if self.training:
            len_out = inputs.size(1) +1
        else:
            len_out = 1
                                   
        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.M)
        q = F.softmax(params_pen).view(len_out,-1,3)
        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,q #,hidden,cell

class StartModel(nn.Module):
    def __init__(self, hp):
        super(StartModel, self).__init__()
        self.enc_hidden_size = hp.enc_hidden_size
        self.M = hp.M
        self.encoder = EncoderCNN()
        self.fc_params = nn.Linear(hp.enc_hidden_size, 6 * hp.M)

    def forward(self, img_input):
        """
        img_input: (B, C, H, W)
        """
        encoder_out = self.encoder(img_input)
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.enc_hidden_size) #(B, H*W, C)
        mean_encoder_out = encoder_out.mean(dim=1) #(B, C)
        y = self.fc_params(mean_encoder_out) #(B, 6*M)

        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params) #(M, B, 6)
        
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2) #(M, B, 1)
  
        pi = F.softmax(pi.transpose(0,1).squeeze(), dim=1).view(-1, self.M) # (B, M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(-1, self.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(-1, self.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(-1, self.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(-1, self.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(-1, self.M)
        
        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy
        

class StrokeModel(nn.Module):

    def __init__(self, hp, Nmax):
        super(StrokeModel, self).__init__()
        self.enc_hidden_size = hp.enc_hidden_size
        self.Nz = hp.Nz
        self.Nmax = Nmax
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(hp.dec_hidden_size, hp.Nz, hp.M, hp.dropout)
        self.init_z = nn.Linear(hp.enc_hidden_size, hp.Nz)

    def forward(self, batch_init, img_input):
        """
        
        """
        encoder_out = self.encoder(img_input)
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.enc_hidden_size) #(B, H*W, C)
        mean_encoder_out = encoder_out.mean(dim=1) #(B,C)
        z = self.init_z(mean_encoder_out)
        z_stack = torch.stack([z]*(self.Nmax+1)) #(Nmax+1, B, z_dim) 
        inputs = torch.cat([batch_init, z_stack], 2) #(Nmax+1, B, z_dim + 5)
        return self.decoder(inputs, z)

def bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    z_x = ((dx-mu_x)/sigma_x)**2 # (129,16,20) - (129,16,20)
    z_y = ((dy-mu_y)/sigma_y)**2
    z_xy = (dx-mu_x)*(dy-mu_y)/(sigma_x*sigma_y)
    z = z_x + z_y -2*rho_xy*z_xy
    exp = torch.exp(-z/(2*(1-rho_xy**2)))
    norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1-rho_xy**2)
    return exp / (1e-5+norm)


def reconstruction_loss(mask, dx, dy, p, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    Nmax, batch_size, _ = dx.shape #(129, 16, 20)
    pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
    LS = -torch.sum(mask*torch.log(1e-5+torch.sum(pi * pdf, 2))) / float(Nmax * batch_size)
    LP = -torch.sum(p*torch.log(q))/float(Nmax * batch_size)
    return LS+LP

def lr_decay(optimizer, hp):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


def train_start(hp):
    train_files, test_files = None,None 
    with open('/raid/xiaoyuz1/startmodel_data.pickle', 'rb') as f:
        train_files, test_files = pickle.load(f)
    dataset = StartDataset(hp.root_dir, train_files)
    model = StartModel(hp)
    model = model.cuda()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), hp.lr)

    for epoch in range(hp.epochs):
        for i, (img_input, start_start_positions) in enumerate(tqdm(train_loader)):
            start_start_positions = start_start_positions.cuda()
            img_input = img_input.cuda()

            batch_size = start_start_positions.size(0)

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = model(img_input)
            
            dx = torch.stack([start_start_positions.data[:,0]] * hp.M, 1) #(B, M)
            dy = torch.stack([start_start_positions.data[:,1]] * hp.M, 1) #(B, M)

            # prepare optimizers:
            optimizer.zero_grad()

            batch_size, _ = dx.shape #(16, 20)
            pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
            # print(pdf.shape, pi.shape)
            # print(torch.sum(pi * pdf, 1))
            # import pdb; pdb.set_trace()

            LS = -torch.sum(torch.log(1e-5 + torch.sum(pi * pdf, 1))) / float(batch_size)
            # print(torch.isnan(LS))
            if torch.isnan(LS).item():
                import pdb; pdb.set_trace()
            loss = LS
            loss.backward()
            # gradient cliping
            nn.utils.clip_grad_norm_(model.encoder.parameters(), hp.grad_clip)
            # optim step
            optimizer.step()

            if i % 100 == 0:
                print('epoch', epoch, ' | i', i,  ' | loss', loss.item())

        # some print and save:
        if epoch % 1 ==0:
            optimizer = lr_decay(optimizer, hp)
            sel = np.random.rand()
            torch.save(model.state_dict(), '/raid/xiaoyuz1/test_stroke_model/encoderCNN_sel_%3f_epoch_%d.pth' % (sel,epoch))


def train_stroke(hp):
    dataset = PartDataset(hp)
    model = StrokeModel(hp, dataset.Nmax)
    model = model.cuda()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), hp.lr)

    for epoch in range(hp.epochs):
        for i, (stroke_input, stroke_lengths, img_input) in enumerate(train_loader):
            stroke_input = stroke_input.cuda()
            img_input = stroke_input.cuda()

            batch_size = stroke_input.size(1)

            if use_cuda:
                sos = torch.stack([torch.Tensor([0,0,1,0,0])] * batch_size).cuda().unsqueeze(0)
            else:
                sos = torch.stack([torch.Tensor([0,0,1,0,0])] * batch_size).unsqueeze(0)
            batch_init = torch.cat([sos, stroke_input],0) #(Nmax+1, B, 5)

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = model(batch_init, stroke_lengths, img_input)
            
            if use_cuda:
                eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch_size).cuda().unsqueeze(0)
            else:
                eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch_size).unsqueeze(0)
            batch = torch.cat([stroke_input, eos], 0) #(129+1, 16, 5)
            mask = torch.zeros(len(stroke_input)+1, batch_size) #(129+1, N)
            for indice, length in enumerate(stroke_lengths):
                mask[:length, indice] = 1
            if use_cuda:
                mask = mask.cuda()
            dx = torch.stack([stroke_input.data[:,:,0]] * hp.M, 2) #(129, 16, 20)
            dy = torch.stack([stroke_input.data[:,:,1]] * hp.M, 2)
            p1 = stroke_input.data[:,:,2]
            p2 = stroke_input.data[:,:,3]
            p3 = stroke_input.data[:,:,4]
            p = torch.stack([p1,p2,p3],2)

            # prepare optimizers:
            optimizer.zero_grad()

            LR = reconstruction_loss(mask, dx, dy, p, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
            loss = LR
            loss.backward()
            # gradient cliping
            nn.utils.clip_grad_norm(model.encoder.parameters(), hp.grad_clip)
            nn.utils.clip_grad_norm(model.decoder.parameters(), hp.grad_clip)
            # optim step
            optimizer.step()

            # some print and save:
            if epoch % 1==0:
                print('epoch', epoch, 'loss', loss.data[0], 'LR', LR.data[0])
                optimizer = lr_decay(optimizer, hp)

                sel = np.random.rand()
                torch.save(model.state_dict(), 'encoderCNN_decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))

if __name__=="__main__":
    hp = HParams()
    train_start(hp)