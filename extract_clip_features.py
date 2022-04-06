import subprocess
import os # adding this line
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/" # adding this line

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd
pydiffvg.set_print_timing(False)
from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import PIL
from torchvision import utils
import numpy as np
import torch
import os
print("Torch version:", torch.__version__)

#@title Load CLIP {vertical-output: true}

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import os
import clip
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

parser = argparse.ArgumentParser(description='extract clip features of images')
parser.add_argument("--folder", type=str, help="folder of images")
parser.add_argument("--feature_save_folder", type=str, help="folder of saved features")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--start_idx", type=int, help="start_idx")
parser.add_argument("--end_idx", type=int, help="end_idx")
parser.add_argument("--flip", type=int, default=0, help="flip the picture")

args = parser.parse_args()

folder = args.folder
sub_files = []
for file in os.listdir(folder):
    if file.endswith(".png"):
        sub_files.append(file)
print("Total number of images: ", len(sub_files))

feature_save_folder = args.feature_save_folder
if not os.path.exists(feature_save_folder):
    os.mkdir(feature_save_folder)

start_idx = int(args.start_idx) #int(input("Start index?"))  # Python 3
end_idx = int(args.end_idx) #int(input("End index?"))  # Python 3

print(start_idx, end_idx)

# Load the model
device = torch.device('cuda')
model, preprocess = clip.load('ViT-B/32', device, jit=False)
resize_to_clip = transforms.Compose([
    transforms.Resize(size=224, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
])

batch_size = args.batch_size

acc = batch_size
all_batches = []
all_image_paths = []

img_augs = []
image_paths = []
count = 0
for f_idx, f in enumerate(sub_files):
    
    if not f.endswith(".png"):
        print("WARNING: encountered ", f)
        continue
    
    if not(f_idx >= start_idx and f_idx < end_idx):
        continue
    
    img_path = os.path.join(folder, f)
    pimg = PIL.Image.open(img_path)
    image_paths.append(img_path)
    if(len(np.asarray(pimg).shape) < 3):
        pimg = pimg.convert(mode='RGB')
    if args.flip:
        import PIL.ImageOps
        pimg = PIL.ImageOps.invert(pimg)
    
    img_augs.append(resize_to_clip(pimg).unsqueeze(0))
    acc -= 1
    count += 1
    if f_idx == len(sub_files)-1:     
        all_batches.append(img_augs)
        all_image_paths.append(image_paths)
        break
    if acc < 1 or f_idx == (end_idx-1):
        acc = batch_size
        all_batches.append(img_augs)
        all_image_paths.append(image_paths)
        img_augs = [] 
        image_paths = []

# print(count)

all_image_features = []
batch_start_idx = start_idx / batch_size

for batch_idx in range(len(all_batches)):
    
    save_prefix = int(batch_start_idx + batch_idx)
    print(save_prefix, batch_idx)
    
    feature_path = os.path.join(feature_save_folder, '{}.npy'.format(save_prefix))
    image_name_path = os.path.join(feature_save_folder, '{}.txt'.format(save_prefix))
    with open(image_name_path, "w+") as f:
        f.write("\n".join(all_image_paths[batch_idx]))
    
    im_batch = torch.cat(all_batches[batch_idx])
    im_batch = im_batch.cuda(pydiffvg.get_device())
    with torch.no_grad():
        image_features = model.encode_image(im_batch)
    all_image_features.append(image_features)
    print(image_features.shape)
    with open(feature_path, 'wb') as f:
        np.save(f, image_features.cpu().numpy())