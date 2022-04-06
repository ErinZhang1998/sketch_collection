import PIL
import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageOps
import read_datasets as rd
import cairocffi as cairo
import os
import json
import collections
import constants as CONST

angel_json = json.load(open(
    '/raid/xiaoyuz1/sketch_datasets/SketchX-PRIS-Dataset/Perceptual Grouping/{}.ndjson'.format('angel'), 
    'r'))

for part_idx,L in CONST.angel_part_image_indices.items():
    if not os.path.exists('/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5/{}'.format(part_idx)):
        os.mkdir('/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5/{}'.format(part_idx))
    
    for i in L:
        vector_part = rd.create_im(angel_json, i, part_idxs=[part_idx])
        path = "/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5/{}/{}.png".format(part_idx, i)
        rd.render_img(vector_part, img_path=path, line_diameter=5)

# entire
if not os.path.exists('/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5_all'):
    os.mkdir('/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5_all')

for part_idx,L in CONST.angel_part_image_indices.items():
    for i in L:
        vector_part = rd.create_im(angel_json, i, part_idxs=[])
        path = "/raid/xiaoyuz1/sketch_datasets/angel_images_weight_5_all/{}.png".format(i)
        rd.render_img(vector_part, img_path=path, line_diameter=5)

# face_json = json.load(open(
#     '/raid/xiaoyuz1/sketch_datasets/SketchX-PRIS-Dataset/Perceptual Grouping/{}.ndjson'.format('face'), 
#     'r'))

# def all_indices_with_parts(L, part_idx):
#     idxs = []
#     for idx in L:
#         drawing_raw = face_json['train_data'][idx]
#         labels = np.unique(np.asarray(drawing_raw)[:,-1])
#         if part_idx in labels:
#             idxs.append(idx)
#     return idxs

# cluster_dir = "/raid/xiaoyuz1/face_cluster_new"
# groups_idx = collections.defaultdict(list)
# for i,c in enumerate(os.listdir(cluster_dir)):
#     L = os.listdir(os.path.join(cluster_dir, c))
#     for p in L:
#         img1 = int(p.split(".")[0])
#         groups_idx[c].append(img1)
    
# face_indices = []
# png_to_group = {}
# for c,L in groups_idx.items():
#     face_indices += L
#     png_to_group.update(dict(zip(L, [c]*len(L))))

# def create_im(i, part_idxs=[]):
#     drawing_raw = face_json['train_data'][i]
#     drawing_new = rd.transform_spg_2_quickdraw(drawing_raw, label_selected=part_idxs)
#     vector_part = []
#     for stroke in drawing_new:
#         stroke = np.asarray(stroke).T
#         vector_part.append(stroke)
#     return vector_part

# for part_idx in [0,1,2,4,6]:
#     if not os.path.exists('/raid/xiaoyuz1/sketch_datasets/face_images_weight_5/{}'.format(part_idx)):
#         os.mkdir('/raid/xiaoyuz1/sketch_datasets/face_images_weight_5/{}'.format(part_idx))
    
#     L = all_indices_with_parts(face_indices, part_idx)
#     for i in L:
#         vector_part = create_im(i, part_idxs=[part_idx])
#         path = "/raid/xiaoyuz1/sketch_datasets/face_images_weight_5/{}/{}.png".format(part_idx, i)
#         rd.render_img(vector_part, img_path=path, line_diameter=5)

# # entire
# if not os.path.exists('/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all'):
#     os.mkdir('/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all')

# for i in face_indices:
#     vector_part = create_im(i, part_idxs=[])
#     path = "/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all/{}.png".format(i)
#     rd.render_img(vector_part, img_path=path, line_diameter=5)