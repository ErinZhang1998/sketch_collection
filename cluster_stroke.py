import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import PIL
import read_datasets as rd
import pickle
import collections
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
# import sys
# sys.path.insert(0, "/home/xiaoyuz1/mnist-em-bmm-gmm")
# import gmm
# import kmeans
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-paths', nargs='+', required=True)
parser.add_argument('-test_paths', nargs='+')
parser.add_argument('-pca_dim', type=int, required=True)
parser.add_argument('-num_cluster', type=int, required=True)
parser.add_argument('-image_format', type=int, required=True)
parser.add_argument('-canvas_size', type=int, default=28)
parser.add_argument('-original_canvas_size', type=float, default=28.)
parser.add_argument('-model_save_path', type=str, required=True)
parser.add_argument('-visualize_save_folder', type=str, required=True)
parser.add_argument('-visualize_path_prefix', type=str)

def view_cluster(features, image_indices, image_format, titles=None, canvas_size=None, original_canvas_size=None, save_path=None, show=False):
    '''
    image_format:
        0: points based images (N, num points, 2)
        1: rendered images (N, image_size, image_size)
        2: image files [N,] in str format
    canvas_size: 
    original_canvas_size
    '''
    canvas_size = int(canvas_size)
    if image_format == 1:
        if canvas_size is None:
            raise ValueError('Must provide canvas_size when passing in images.')
    
    if image_format == 0:
        if canvas_size is None or original_canvas_size is None:
            raise ValueError('Must provide canvas_size and original_canvas_size when passing images that are point based.')
    
    fig = plt.figure(figsize = (25,75))
    fig.patch.set_alpha(1) 
    
    N = len(image_indices)
    file_indices = image_indices

    if N > 100:
        file_indices = np.random.choice(image_indices, 100)
    
    for index, file_idx in enumerate(file_indices):
        plt.subplot(20,5,index+1);
        if image_format == 2:
            file = features[file_idx]
            img = PIL.Image.open(file)
        elif image_format == 1:
            arr = features[file_idx].astype(np.uint8).reshape((canvas_size, canvas_size))
            img = PIL.Image.fromarray(arr)
        elif image_format == 0:
            image = features[file_idx].reshape((-1,2))
            img = rd.render_img(
                [image],
                side=canvas_size,
                line_diameter=1,
                original_side = original_canvas_size,
            )
        else:
            raise ValueError('image_format can only be [0,1,2] but provided {}'.format(image_format))
        
        img = img.convert(mode='RGB')
        img = np.array(img)
        plt.imshow(img)
        if titles is not None:
            plt.title(titles[index])
        plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def read_file(path):
    '''
    read from path and return float numpy array of (num_samples, feat_dim)
    '''
    if path.endswith(".npy"):
        feat = np.load(path)
    elif path.endswith(".npz"):
        try:
            feat = np.load(path, allow_pickle=True,encoding='latin1')['data']    
        except:
            feat = np.load(path, allow_pickle=True,encoding='latin1')['train']
    N = len(feat)
    feat = feat.reshape((N,-1))
    return feat

def read_files(paths):
    feature_list = []
    for path in paths:
        arr = read_file(path)
        feature_list.append(arr)
    features = np.vstack(feature_list)
    print("Feature array dimension: ", features.shape)
    return features

def evaluate(args, model, reducer):
    features = read_files(args.test_paths)
    features_binary = np.where(features > 254, 0, 1)
    print("Predict for test images...")
    features_reduced = reducer.transform(features_binary)
    labels = model.predict(features_reduced)
    clusters = collections.defaultdict(list)
    for image_idx, cluster_idx in zip(np.arange(len(features)), labels):
        clusters[cluster_idx].append(image_idx)
    
    save_folder = args.visualize_save_folder+"-TEST"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    print("Saving evaluation cluster...")
    for cluster_idx in tqdm(np.unique(model.labels_)):
        image_indices = clusters[cluster_idx]
        if args.visualize_path_prefix is not None:
            save_path = os.path.join(save_folder, f'{args.visualize_path_prefix}_{cluster_idx}.png')
        else:
            save_path = os.path.join(save_folder, f'{cluster_idx}.png')
        view_cluster(features, image_indices, 1, canvas_size=args.canvas_size, original_canvas_size=args.original_canvas_size, save_path=save_path)
    

def main(args):
    feature_list = []
    for path in args.paths:
        arr = read_file(path)
        feature_list.append(arr)
    features = np.vstack(feature_list)
    print("Feature array dimension: ", features.shape)
    features_binary = np.where(features > 254, 0, 1)
    if args.pca_dim >= 0:
        reducer = PCA(n_components=args.pca_dim, random_state=22)
        reducer.fit(features_binary)
        features_reduced = reducer.transform(features_binary)
    else:
        features_reduced = features_binary

    print("Calculating clusters...")
    start_time = time.time()
    model = KMeans(n_clusters=args.num_cluster, random_state=22)
    model.fit(features_reduced)
    print("[DONE] KMeans:", "--- %s seconds ---" % (time.time() - start_time))
    
    clusters = collections.defaultdict(list)
    for image_idx, cluster_idx in zip(np.arange(len(features)), model.labels_):
        clusters[cluster_idx].append(image_idx)
    
    with open(args.model_save_path, "wb+") as f:
        pickle.dump(model, f)
        
    if not os.path.exists(args.visualize_save_folder):
        os.mkdir(args.visualize_save_folder)
    
    for cluster_idx in tqdm(np.unique(model.labels_)):
        image_indices = clusters[cluster_idx]
        if args.visualize_path_prefix is not None:
            save_path = os.path.join(args.visualize_save_folder, f'{args.visualize_path_prefix}_{cluster_idx}.png')
        else:
            save_path = os.path.join(args.visualize_save_folder, f'{cluster_idx}.png')
        view_cluster(features, image_indices, args.image_format, canvas_size=args.canvas_size, original_canvas_size=args.original_canvas_size, save_path=save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)