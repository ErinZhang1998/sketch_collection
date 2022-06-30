import read_datasets as rd
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-path', type=str, help='npz or json file path of drawings', required=True) 
parser.add_argument('-start_index', type=int)
parser.add_argument('-end_index', type=int)
parser.add_argument('-side_length', type=int, required=True)
parser.add_argument('-b_spline_degree', type=int, required=True)
parser.add_argument('-b_spline_num_sampled_points', type=int, required=True)
parser.add_argument('-save_prefix', type=str, required=True)
parser.add_argument('-json_file', action='store_true')

parser.add_argument('-process_to_image', action='store_true')
parser.add_argument('-image_length', type=int, default=28)


def process_to_image(stroke_data, image_length, original_length, save_path):
    all_stroke_data = np.vstack(stroke_data)
    all_stroke_img = []
    for i in tqdm(range(len(all_stroke_data))):
        stroke = all_stroke_data[i]
        stroke_image = rd.render_img(
            [stroke], 
            show=False,
            side=int(image_length),
            line_diameter=1,
            padding=0,
            bg_color=(0,0,0),
            fg_color=(1,1,1),
            original_side = original_length,
            convert = True,
        )
        arr = np.array(stroke_image)
        all_stroke_img.append(arr)
    np.savez_compressed(save_path, data=np.asarray(all_stroke_img))

def main(args):
    if args.json_file:
        drawing_arr = json.load(open(args.path, 'r'))['train_data']
    else:
        drawing_npz = np.load(args.path, allow_pickle=True,encoding='latin1')
        drawing_arr = drawing_npz['train']
    
    all_stroke_data = []
    start_index = args.start_index if args.start_index is not None else 0
    end_index = args.end_index if args.end_index is not None else len(drawing_arr)
    
    for i in tqdm(range(start_index, end_index)):
        drawing_raw = np.asarray(drawing_arr[i])
        strokes = rd.process_quickdraw_to_stroke(drawing_raw, side=args.side_length, b_spline_degree=args.b_spline_degree, b_spline_num_sampled_points=args.b_spline_num_sampled_points)
        all_stroke_data.append(np.asarray(strokes))

    np.savez_compressed('/raid/xiaoyuz1/{}-{}-{}-{}-{}-{}'.format(args.save_prefix, args.side_length, args.b_spline_degree,args.b_spline_num_sampled_points,start_index, end_index), train=np.asarray(all_stroke_data, dtype=object))
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)