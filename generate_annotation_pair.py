from tsnecuda import TSNE
import numpy as np
import os
import PIL.ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

feature_folder_template = "/raid/xiaoyuz1/sketch_datasets/face_features/clip/{}"
flip = False

data_dict = {
    0 : (256, 16),

}

def generate_pair(part_idx):
    feature_folder = feature_folder_template.format(str(part_idx))
    pca_dim, num_clusters = data_dict[part_idx]
    
    all_features = []
    all_images = []

    for file in os.listdir(feature_folder):
        if not file.endswith(".npy"):
            continue
            
        img_path = "{}.txt".format(file.split(".n")[0])
        feat = np.load(os.path.join(feature_folder, file))
        image_paths = None
        with open(os.path.join(feature_folder, img_path), 'r') as f:
            lines = f.readlines()
            image_paths = [line.strip() for line in lines]
            #all_images += image_paths
        #print(len(image_paths))
            
        for feati,pathi in zip(feat,image_paths):
            #if pathi.split("/")[-1] in L:
            all_features += [feati]
            all_images += [pathi]
    
    features = np.vstack(all_features)
    print(features.shape)
    pca = PCA(n_components=pca_dim, random_state=22)
    pca.fit(features)
    x = pca.transform(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    kmeans.fit(x)