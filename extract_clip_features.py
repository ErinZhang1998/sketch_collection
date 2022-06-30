import subprocess
import os # adding this line
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/" # adding this line

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import pydiffvg
import torch
import os 
import clip
import argparse
import torchvision.transforms as transforms
pydiffvg.set_print_timing(False)
import PIL
import numpy as np
import PIL.ImageOps
from tqdm import tqdm
print("Torch version:", torch.__version__)

parser = argparse.ArgumentParser(description='extract clip features of images')
parser.add_argument("--folder", type=str, help="folder of images to extract CLIP features for")
# parser.add_argument("--png_file", type=str, help="if not providing a folder of images, we need to have a npz file of the images")
parser.add_argument("--feature_save_folder", type=str, help="folder of saved features")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--start_idx", type=int, help="start_idx")
parser.add_argument("--end_idx", type=int, help="end_idx")
parser.add_argument("--flip", type=int, default=0, help="flip the picture")

def extract_feature(model, images, image_names, feature_save_folder, save_prefix):    
    feature_path = os.path.join(feature_save_folder, '{}.npy'.format(save_prefix))
    image_name_path = os.path.join(feature_save_folder, '{}.txt'.format(save_prefix))
    
    im_batch = torch.cat(images)
    im_batch = im_batch.cuda(pydiffvg.get_device())
    with torch.no_grad():
        image_features = model.encode_image(im_batch)
    with open(feature_path, 'wb') as f:
        np.save(f, image_features.cpu().numpy())
    with open(image_name_path, "w+") as f:
        f.write("\n".join(image_names))

def main(args):
    # collect all png for CLIP feature extraction
    folder = args.folder
    batch_size = args.batch_size
    sub_files = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            sub_files.append(file)
    sub_files = sorted(sub_files)
    total_num_images = len(sub_files)
    print("Total number of images: ", total_num_images)

    # create output folder
    feature_save_folder = args.feature_save_folder
    if not os.path.exists(feature_save_folder):
        os.mkdir(feature_save_folder)

    start_idx = int(args.start_idx) if args.start_idx is not None else 0
    end_idx = int(args.end_idx) if args.end_idx is not None else total_num_images
    print("Start and End index: ", start_idx, end_idx)

    # Load the model
    device = torch.device('cuda')
    model, preprocess = clip.load('ViT-B/32', device, jit=False)
    resize_to_clip = transforms.Compose([
        transforms.Resize(size=224, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
    ])
    batch_start_idx = start_idx / batch_size
    acc = batch_size
    batch_idx = 0

    img_augs = []
    image_paths = []
    for f_idx in tqdm(range(total_num_images)):
        
        f = sub_files[f_idx]
        if not f.endswith(".png"):
            print("WARNING: encountered file not ending in png", f)
            continue
        
        if not(f_idx >= start_idx and f_idx < end_idx):
            continue
        
        img_path = os.path.join(folder, f)
        pimg = PIL.Image.open(img_path)
        
        if(len(np.asarray(pimg).shape) < 3):
            pimg = pimg.convert(mode='RGB')
        if args.flip:
            pimg = PIL.ImageOps.invert(pimg)
        
        img_augs.append(resize_to_clip(pimg).unsqueeze(0))
        image_paths.append(img_path)
        acc -= 1
        
        # if f_idx == len(sub_files)-1:     
        #     break
        
        if acc < 1 or f_idx == (end_idx-1):
            save_prefix = int(batch_start_idx+batch_idx)
            extract_feature(model, img_augs, image_paths, feature_save_folder, save_prefix)
            batch_idx += 1
            acc = batch_size
            img_augs = [] 
            image_paths = []
        
        if f_idx == total_num_images-1:
            break

    # for batch_idx in range(len(all_batches)):
    #     save_prefix = int(batch_start_idx + batch_idx)
    #     # print(save_prefix, batch_idx)
        
    #     feature_path = os.path.join(feature_save_folder, '{}.npy'.format(save_prefix))
    #     image_name_path = os.path.join(feature_save_folder, '{}.txt'.format(save_prefix))
    #     with open(image_name_path, "w+") as f:
    #         f.write("\n".join(all_image_paths[batch_idx]))
        
    #     im_batch = torch.cat(all_batches[batch_idx])
    #     im_batch = im_batch.cuda(pydiffvg.get_device())
    #     with torch.no_grad():
    #         image_features = model.encode_image(im_batch)
    #     all_image_features.append(image_features)
    #     with open(feature_path, 'wb') as f:
    #         np.save(f, image_features.cpu().numpy())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)