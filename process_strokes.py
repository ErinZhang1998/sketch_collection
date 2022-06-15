import read_datasets as rd
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-npz_path', type=str, help='npz file path of drawings', required=True) # '/raid/xiaoyuz1/sketch_datasets/sketchrnn_face.npz'
parser.add_argument('-start_index', type=int)
parser.add_argument('-end_index', type=int)
parser.add_argument('-side_length', type=int, required=True)

parser.add_argument('-save_prefix', type=str, required=True)

def main(args):
    drawing_npz = np.load(
        args.npz_path, 
        allow_pickle=True,
        encoding='latin1',
    )
    drawing_arr = drawing_npz['train']
    
    all_stroke_data = []
    start_index = args.start_index if args.start_index is not None else 0
    end_index = args.end_index if args.end_index is not None else len(drawing_arr)
    for i in tqdm(range(start_index, end_index)):
        drawing_raw = drawing_arr[i]
        strokes = rd.process_quickdraw_to_stroke(drawing_raw, side=args.side_length)
        all_stroke_data.append(np.asarray(strokes))

    np.savez_compressed('/raid/xiaoyuz1/{}_{}_{}_{}'.format(args.save_prefix, args.side_length, start_index, end_index), train=np.asarray(all_stroke_data, dtype=object))
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)