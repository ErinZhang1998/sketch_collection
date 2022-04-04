import json
import os
import pandas as pd
import pickle 

parts_idx = [0,1,2,4,6]
parts_idx_dict = {
    0 : "eyes",
    1: "nose",
    2: "mouth",
    4: "hair",
    6: "outline of face",
}

part_idx_to_words = {
    0 : "eyes",
    1 : "nose",
    2 : "mouth",
    4 : "hair",
    6 : "face",
}

face_json = json.load(open(
    '/raid/xiaoyuz1/sketch_datasets/SketchX-PRIS-Dataset/Perceptual Grouping/{}.ndjson'.format('face'), 
    'r'))

spg_face_folder = '/raid/xiaoyuz1/sketch_datasets/spg/face'
spg_angel_folder = '/raid/xiaoyuz1/sketch_datasets/spg/angel'
all_face_indices = [int(f.split(".")[0]) for f in os.listdir('/raid/xiaoyuz1/sketch_datasets/face_images_weight_5_all')]
missing_face_indices = []
for i in range(800):
    if i not in all_face_indices:
        missing_face_indices.append(i)
face_part_image_indices = None
with open('/raid/xiaoyuz1/face_part_image_indices.pickle', 'rb') as f:
    face_part_image_indices = pickle.load(f)

face_part1_df = pd.read_csv('/raid/xiaoyuz1/amazon_turk/df_all_pair.csv')
face_part1_df['no_punc_1'] = face_part1_df.no_punc_1.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])
face_part1_df['no_punc_2'] = face_part1_df.no_punc_2.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])