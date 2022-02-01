import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import PIL
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/xiaoyuz1/multigraph_transformer")
from dataloader.QuickdrawDataset import *

from utils.AverageMeter import AverageMeter
from utils.accuracy import *
import argparse


parser = argparse.ArgumentParser(description='extract clip features of images')
parser.add_argument("--folder", type=str, help="folder of images")
parser.add_argument("--label", type=str, help="folder of images")

parser.add_argument("--feature_save_folder", type=str, help="folder of saved features")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--start_idx", type=int, help="start_idx")
parser.add_argument("--end_idx", type=int, help="end_idx")
parser.add_argument("--flip", type=int, default=0, help="flip the picture")

args = parser.parse_args()

folder = args.folder #"/raid/xiaoyuz1/sketch_datasets/spg/airplane"
sub_files = []
for file in os.listdir(folder):
    if file.endswith(".png"):
        sub_files.append(file)
print("Total number of images: ", len(sub_files))
flip = args.flip == 1

cnn_train_data_labels = {}
with open('/raid/xiaoyuz1/sketch_datasets/data_4_cnnbaselines/tiny_train_set.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        k = line.split(" ")[0].split("/")[0]
        v = int(line.split(" ")[1])
        cnn_train_data_labels[k] = v
one_label = cnn_train_data_labels[args.label]
print(one_label)
sketch_list_fname = os.path.join(folder, 'annotation.txt')
with open(sketch_list_fname, 'w+') as f:
    for file in sub_files:
        if not file.endswith(".png"):
            continue
        line = '{} {}\n'.format(file, one_label)
        f.write(line)

transform_val = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor()
])
feat_dataset = QuickdrawDataset(
    folder, 
    sketch_list_fname, 
    transform_val,
    flip=flip)

feat_loader = DataLoader(
    feat_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=12)

model_ckpt_folder = os.path.join(
    "/home/xiaoyuz1/multigraph_transformer/baselines/cnn_baselines",
    "experimental_results/inceptionv3_001/checkpoints"
)
model_path = os.path.join(model_ckpt_folder, 'inceptionv3_001_net_epoch17')
model = models.inception_v3(num_classes = 345)
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path)['network']

for k in model_dict.keys():
    if k not in pretrained_dict:
        print(k)

pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(pretrained_dict)

class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        
        self.my_Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.my_Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.my_Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.my_maxpool1 = inception.maxpool1
        self.my_Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.my_Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.my_maxpool2 = inception.maxpool2
        self.my_Mixed_5b = inception.Mixed_5b
        self.my_Mixed_5c = inception.Mixed_5c
        self.my_Mixed_5d = inception.Mixed_5d
        self.my_Mixed_6a = inception.Mixed_6a
        self.my_Mixed_6b = inception.Mixed_6b
        self.my_Mixed_6c = inception.Mixed_6c
        self.my_Mixed_6d = inception.Mixed_6d
        self.my_Mixed_6e = inception.Mixed_6e
        self.my_AuxLogits = inception.AuxLogits
        self.my_Mixed_7a = inception.Mixed_7a
        self.my_Mixed_7b = inception.Mixed_7b
        self.my_Mixed_7c = inception.Mixed_7c
        
        self.avgpool = inception.avgpool
        self.fc = inception.fc

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        
        x = self.my_Conv2d_1a_3x3(x)
        x = self.my_Conv2d_2a_3x3(x)
        x = self.my_Conv2d_2b_3x3(x)
        x = self.my_maxpool1(x)
        x = self.my_Conv2d_3b_1x1(x)
        x = self.my_Conv2d_4a_3x3(x)
        x = self.my_maxpool2(x)
        
        x = self.my_Mixed_5b(x)
        x = self.my_Mixed_5c(x)
        x = self.my_Mixed_5d(x)
        
        x = self.my_Mixed_6a(x) #768, 17, 17
        x = self.my_Mixed_6b(x)
        x = self.my_Mixed_6c(x)
        x = self.my_Mixed_6d(x)
        x = self.my_Mixed_6e(x) #768, 17, 17
        
#         x = self.my_AuxLogits(x)
        
        x = self.my_Mixed_7a(x)
        x = self.my_Mixed_7b(x)
        x = self.my_Mixed_7c(x)
        
        x = self.avgpool(x)
        
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(x)
#         # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x)
#         # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x)
#         # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x)
#         # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x)
#         # 71 x 71 x 192
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 35 x 35 x 192
#         x = self.Mixed_5b(x)
#         # copy paste from model definition, just stopping where you want
        return x

net = MyInceptionFeatureExtractor(model)
net.cuda()
net.eval()


validation_acc = AverageMeter()

extracted_features = []
with torch.no_grad():
    for idx, (sketch, label) in enumerate(tqdm(feat_loader, ascii=True)):

        sketch = sketch.cuda()
        label = label.cuda()

        output = net(sketch)
        output = output.squeeze()
        
        print(accuracy(net.fc(output), label, topk = (1,))[0].item(), sketch.size(0))
        validation_acc.update(accuracy(net.fc(output), label, topk = (1,))[0].item(), sketch.size(0))
        extracted_features.append(output.detach().cpu().numpy())
#         print(output.shape)
#         break

print(validation_acc.avg)
feature_save_folder = args.feature_save_folder
if not os.path.exists(feature_save_folder):
    os.mkdir(feature_save_folder)

save_prefix = 0
feature_path = os.path.join(feature_save_folder, '{}.npy'.format(save_prefix))
image_name_path = os.path.join(feature_save_folder, '{}.txt'.format(save_prefix))

with open(image_name_path, "w+") as f:
    f.write("\n".join(feat_loader.dataset.sketch_urls))

with open(feature_path, 'wb') as f:
    np.save(f, np.vstack(extracted_features))