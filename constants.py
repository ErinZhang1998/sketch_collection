import json
import os
import pandas as pd
import pickle 

def loading_quickdraw_npz(is_face = True, face_path = '/raid/xiaoyuz1/sketch_datasets/sketchrnn_face.npz', angel_path = '/raid/xiaoyuz1/sketch_datasets/sketchrnn_angel.npz'):
    if is_face:
        drawing_npz = np.load(face_path, allow_pickle=True,encoding='latin1')
    else:
        drawing_npz = np.load(angel_path, allow_pickle=True,encoding='latin1')
    return drawing_npz

face_parts_idx = [0,1,2,4,6]
face_parts_idx_dict = {
    0: "eyes",
    1: "nose",
    2: "mouth",
    3: "ear",
    4: "hair",
    5: "moustache",
    6: "outline of face",
}
face_parts_idx_dict_doodler = {
    0: "eyes",
    1: "nose",
    2: "mouth",
    4: "hair",
    6: "face",
}
# part_idx_to_words = {
#     0 : "eyes",
#     1 : "nose",
#     2 : "mouth",
#     4 : "hair",
#     6 : "face",
# }

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

# face_part1_df = pd.read_csv('/raid/xiaoyuz1/amazon_turk/df_all_pair.csv')
# face_part1_df['no_punc_1'] = face_part1_df.no_punc_1.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])
# face_part1_df['no_punc_2'] = face_part1_df.no_punc_2.apply(lambda x: [str(y).strip()[1:-1] for y in x[1:-1].split(',')])


angel_json = json.load(open(
    '/raid/xiaoyuz1/sketch_datasets/SketchX-PRIS-Dataset/Perceptual Grouping/{}.ndjson'.format('angel'), 
    'r'))

angel_parts_idx = [0,1,2,3,4,5,7]
angel_parts_idx_dict = {
    0:"halo",1 : "eyes",2:"nose",3:"mouth",4:"outline of face",5:"body", 6: "torso", 7:"wings",
}

angel_parts_idx_dict_doodler = {0:"halo", 1 : "eyes", 2:"nose", 3:"mouth", 4:"face", 5:"body", 7:"wings"}

angel_part_image_indices = None
with open('/raid/xiaoyuz1/angel_part_image_indices.pickle', 'rb') as f:
    angel_part_image_indices = pickle.load(f)

all_angel_indices = set()
for k,v in angel_part_image_indices.items():
    for img_idx in v:
        all_angel_indices.add(img_idx)
all_angel_indices = list(all_angel_indices)