import numpy as np
import pandas as pd
import seaborn as sns
import os
import json
import matplotlib.pyplot as plt
import PIL
import pickle 
import scipy.io as scio
import svgwrite
from cairosvg import svg2png
import cairocffi as cairo
import os
import cv2
import pdb
import PIL.ImageOps
import collections
import constants as CONST

def get_features(feature_folder_template):
    
    feature_dict = collections.defaultdict(lambda: {})
    for part_idx in [0,1,2,4,6]:
        feature_folder = feature_folder_template.format(str(part_idx))
        for file in os.listdir(feature_folder):
            if not file.endswith(".npy"):
                continue

            img_path = "{}.txt".format(file.split(".n")[0])
            feat = np.load(os.path.join(feature_folder, file))
            image_paths = None
            with open(os.path.join(feature_folder, img_path), 'r') as f:
                lines = f.readlines()
                image_paths = [line.strip() for line in lines]
            
            for feati,pathi in zip(feat,image_paths):
                feature_dict[part_idx][int(pathi.split("/")[-1].split(".")[0])] = feati
    return feature_dict

def render_img(
    vector_part, 
    img_path=None, 
    show=False,
    side=256,
    line_diameter=10,
    padding=0,
    bg_color=(0,0,0),
    fg_color=(1,1,1),
    original_side = 256.,
):

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)
    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)
    raster_images = []
    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()
    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    for stroke in vector_part:
        if len(stroke) == 0:
            continue
        ctx.move_to(stroke[0][0], stroke[0][1])
        for x, y in stroke:
            ctx.line_to(x, y)
        ctx.stroke()
    surface_data = surface.get_data()
    raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)
    image = PIL.Image.fromarray(raster_image).convert("L")
    image = PIL.ImageOps.invert(image)
    if img_path is not None:
        image.save(img_path)
    if show:
        arr = np.asarray(image)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return image


def data2abspoints(data, label_selected=[]):
    color_map = {
        0 : (255,0,0), #red
        1 : (0,255,0), #green
        2 : (0,0,255), #blue
        3 : (255,255,0), #yellow
        4 : (0,255,255), #cyan
        5 : (255,0,255), #magenta
        6 : (255,192,203), #pink
        # 7 : (128,128,0), #olive
        7: (255,165,0), #orange
        8 : (0,128,128),
        9 : (255,192,203),
        10 : (0,255,0),
    }
    for k,v in color_map.items():
        r,g,b = v
        color_map[k] = ((r/255) * 100, (g/255) * 100, (b/255) * 100)
    
    stroke_color = []
    abspoints = []
    stroke_group = []
    stroke_group_idx = -1
    abs_x = 25
    abs_y = 25
    next_is_gap = 1 #if it is the start of the stroke, then begin and end to itself, otherwise begin from the last point, end at the current point.
    for line_data in data:

        offset_x = line_data[0]
        offset_y = line_data[1]
        label = line_data[-1]
        if next_is_gap:
            begin_point = [abs_x+offset_x,abs_y+offset_y]
            stroke_group_idx+=1
        else:
            begin_point = [abs_x,abs_y]
        end_point = [abs_x+offset_x,abs_y+offset_y]
        abs_x +=offset_x
        abs_y +=offset_y
        next_is_gap = line_data[2]
        if len(label_selected) > 0:
            if label not in label_selected:
                continue
        abspoints.append([begin_point,end_point])
        stroke_color.append(color_map[line_data[3]])
        stroke_group.append(label)

    return abspoints, stroke_group, stroke_color

def to_pngs_other(abspoints, stroke_group, output_dim=(256,256), color=None, stroke_width=1, color_selected=[]):
    stroke_group = np.asarray(stroke_group)
    unique_stroke_group = np.unique(stroke_group)

    stroke_svg_name = 'pic.svg'
    stroke_png_name = 'pic.png'

    dwg = svgwrite.Drawing(stroke_svg_name, size=output_dim)
    dwg.add(dwg.rect(insert=(0, 0), size=output_dim, fill='white'))

    for group_idx in unique_stroke_group:
        same_group_stroke_idxs = np.where(stroke_group==group_idx)[0]

        for idx in same_group_stroke_idxs:
            if max(same_group_stroke_idxs) >= len(abspoints):
                pdb.set_trace()
            begin_point = abspoints[idx][0]
            end_point = abspoints[idx][1]
            if color is not None:
                r,g,b = color[idx]
            else:
                r,g,b = 0,0,0
                        
            if len(color_selected) > 0 and group_idx not in color_selected:
            
                r,g,b = ((211/255) * 100, (211/255) * 100, (211/255) * 100)
            # print(r,g,b)
            dwg.add(
                dwg.line(
                    (begin_point[0], begin_point[1]), 
                    (end_point[0], end_point[1]),
                    stroke = svgwrite.rgb(b,g,r, '%'),
                    stroke_width = stroke_width,
                )
            )
    dwg.save()
    svg2png(url=stroke_svg_name, write_to=stroke_png_name)
    os.remove(stroke_svg_name)
    image_data = cv2.imread(stroke_png_name)
    
    return image_data

def transform_spg_2_svg_png(drawing_raw, output_dim=(256,256), draw_color=False, label_selected=[], stroke_width=1, color_selected=[]):
    abspoints, stroke_group, color = data2abspoints(drawing_raw, label_selected)
    if draw_color:
        return to_pngs_other(abspoints, stroke_group, output_dim, color=color, stroke_width = stroke_width, color_selected=color_selected)
    else:
        return to_pngs_other(abspoints, stroke_group, output_dim, color=None, stroke_width = stroke_width, color_selected=color_selected)

def show_these_sketches(
    obj,
    png_indices, 
    titles, 
    part_indices, 
    show_title=True,
    num_pngs_per_row = 2,
    row_figsize = 6,
    column_figsize = 3,
):
    
    
    num_rows = len(png_indices) // num_pngs_per_row
    if num_rows * num_pngs_per_row < len(png_indices):
        num_rows += 1

    fig = plt.figure(figsize=(num_pngs_per_row * row_figsize, num_rows * column_figsize)) 
    fig.patch.set_alpha(1)  # solution

    for index, idx in enumerate(png_indices):
        plt.subplot(num_rows, num_pngs_per_row, index+1)
        
        drawing_raw = obj['train_data'][idx]
        image_data = transform_spg_2_svg_png(
            drawing_raw, 
            draw_color = True, 
            stroke_width = 3,
            color_selected = part_indices[index],
        )
        
        plt.imshow(image_data)
        plt.title(titles[index])
        plt.axis('off')

    plt.show()
    plt.close()

def to_absolute(drawing_raw):
    '''
    Original format is the relative difference, now transform all to absolute coordinate.
    Canvas size (256,256)
    Shift is (25,25).
    '''
    drawing_new = []
    x,y = 25,25
    for dx,dy,p,l in drawing_raw:
        drawing_new.append([x+dx,y+dy,p,l])
        x,y = x+dx,y+dy
    return drawing_new

def select_absolutedata(data, label_selected=[]):
    ''' ** '''
    result = []
    for data_idx,(x,y,p,l) in enumerate(data):
        
        add = False
        if len(label_selected) == 0 or l in label_selected:
            result.append([x,y,p])
            add = True
        
        if len(label_selected) > 0 and add and len(result) > 1:
            if data[data_idx-1][-1] != l or data[data_idx-1][-2] == 1:
                # ddx,ddy,_,_,_ = result[-2]
                result[-2][2] = 1

    return result

def split_into_individual_strokes(data):
    result = [[[],[]]]
    for data_idx,(x,y,p) in enumerate(data):
        result[-1][0].append(x)
        result[-1][1].append(y)
        if p == 1 and data_idx < len(data)-1:
            result.append([[],[]])
    return result

def show_these_sketches(
    png_indices, 
    titles, 
    part_indices, 
    show_title=True,
    num_pngs_per_row = 2,
    row_figsize = 6,
    column_figsize = 3,
):
    
    
    num_rows = len(png_indices) // num_pngs_per_row
    if num_rows * num_pngs_per_row < len(png_indices):
        num_rows += 1

    fig = plt.figure(figsize=(num_pngs_per_row * row_figsize, num_rows * column_figsize)) 
    fig.patch.set_alpha(1)  # solution

    for index, idx in enumerate(png_indices):
        plt.subplot(num_rows, num_pngs_per_row, index+1)
        
        drawing_raw = CONST.face_json['train_data'][idx]
        image_data = transform_spg_2_svg_png(
            drawing_raw, 
            draw_color = True, 
            stroke_width = 3,
            color_selected = part_indices[index],
        )
        
        plt.imshow(image_data)
        plt.title(titles[index])
        plt.axis('off')

    plt.show()
    plt.close()

# ------------------------------------------------------------------------------------------------
def get_img(img_path):
    img = PIL.Image.open(img_path)
    return np.asarray(img)

def show_images(img_paths, img_titles, flip=False):
    '''
    Read a list of image paths, display 4 in a row.
    '''
    num_images = len(img_paths)
    num_plots_per_row = 5
    num_rows = int(np.ceil(num_images / num_plots_per_row))
    plt.figure(figsize=(num_plots_per_row * 3, num_rows * 3,))

    plot_idx_acc = 1

    for img_path,img_title in zip(img_paths, img_titles):
        img = PIL.Image.open(img_path)
        img = img.convert('RGB')
        if flip:
            img = PIL.ImageOps.invert(img)

        ax = plt.subplot(num_rows, num_plots_per_row , plot_idx_acc)
        plt.imshow(img)
        ax.title.set_text(
            img_title,
        )
        ax.title.set_fontsize(7)
        plot_idx_acc += 1        

    plt.show()

    # for c in clusters:
    # fig = plt.figure(figsize = (25,50))
    # fig.patch.set_alpha(1)  # solution

    # for index, j in enumerate(dir_to_png[c]):
    #     img_idx = int(j.split(".")[0])      
    #     plt.subplot(20,20,index+1)
    #     file = '/raid/xiaoyuz1/sketch_datasets/spg/face/{}.png'.format(img_idx)
    #     img = PIL.Image.open(file)
    #     img = img.convert(mode='RGB')
        
    #     img = np.array(img)
    #     plt.imshow(img)
    #     if index == 0:
    #         plt.title("({}) {}".format(c, j))
    #     else:
    #         plt.title(j)
    #     plt.axis('off')
    # plt.show()
    # plt.close()

def transform_multigraph_2_quickdraw(coord, pen_state):
    '''
    QuickDraw Fromat: [[[xs_stroke_1],[ys_stroke_1]], [[xs_stroke_2],[ys_stroke_2]], ...]

    Dataset: /raid/xiaoyuz1/sketch_datasets/multi_graph_data/tiny_train_dataset_dict.pickle
    '''
    drawing = [[[],[]]]
    for idx, (x,y) in enumerate(coord):
        p = pen_state[idx]
        drawing[-1][0].append(x)
        drawing[-1][1].append(y)
        
        if(p == 102):
            break
        
        if(p == 101):
            drawing.append([[],[]])
        
        
    return drawing

def draw_to_img(drawing, x_lim=256, y_lim=256, save_path=None):
    components_accumatlor = []
    for stroke in drawing:
        stroke = np.asarray(stroke).T
        components_accumatlor += [stroke]

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(0,x_lim)
    plt.ylim(y_lim,0)
    for stroke_idx, part_xy_i in enumerate(components_accumatlor):
        part_xy_i = np.asarray(part_xy_i)
        plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0))
    plt.axis('off')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_per_stroke(drawing, x_lim=256, y_lim=256, save_path=None):
    '''
    drawing is in QuickDraw Format
    '''
    hue_palette = sns.color_palette("husl", 20)
    total_steps = len(drawing)

    num_plots_per_row = 3
    num_rows = int(np.ceil(total_steps / num_plots_per_row))
    plot_idx_acc = 1
    fig = plt.figure(figsize=(num_plots_per_row * 5, num_rows * 5,))

    components_accumatlor = []
    for stroke in drawing:
        stroke = np.asarray(stroke).T
        
        plt.subplot(num_rows, num_plots_per_row , plot_idx_acc)
        plt.xlim(0,x_lim)
        plt.ylim(y_lim,0)

        components_accumatlor += [stroke]

        for stroke_idx, part_xy_i in enumerate(components_accumatlor):
            part_xy_i = np.asarray(part_xy_i)

            if stroke_idx == len(components_accumatlor) - 1:
                plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0))
            else:
                plt.plot(part_xy_i[:,0], part_xy_i[:,1], c='grey')

        plot_idx_acc += 1

    plt.show()
    
    fig = plt.figure(figsize=(5,5))
    plt.xlim(0,x_lim)
    plt.ylim(y_lim,0)
    for stroke_idx, part_xy_i in enumerate(components_accumatlor):
        part_xy_i = np.asarray(part_xy_i)
        plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0))
    plt.axis('off')
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()

def get_rough_match_to_quickdraw_categories(labels, all_categories):
    labels = np.asarray(labels)
    label_map = {}
    for lab in labels:
        L = []
        if lab in all_categories:
            L.append(lab)
        for c in all_categories:
            if lab == c:
                continue
            if lab in c:
                L.append(c)
            else:
                if c in lab:
                    L.append(c)
        label_map[lab] = L
    return label_map

def quickdraw2abspoints(data):
    abspoints = []
    stroke_group = []

    for stroke_group_idx, stroke in enumerate(data):
        stroke = np.asarray(stroke).astype(float).T
        for stroke_idx,(x,y) in enumerate(stroke):
            if stroke_idx == 0:
                begin_point = [x,y]
            else:
                begin_point = [stroke[stroke_idx-1][0],stroke[stroke_idx-1][1]]
            end_point = [x,y]
            abspoints.append([begin_point,end_point])
            stroke_group.append(stroke_group_idx)
    return abspoints, stroke_group

# ------------ SketchSeg
class SketchSeg():
    def __init__(self):
        self.categories = ['airplane', 'bicycle', 'candelabra', 'chair', 'fourleg', 'human', 'lamp', 'rifle', 'table', 'vase']
        self.categories_to_quickdraw = {'airplane': ['airplane'],
        'bicycle': ['bicycle'],
        'candelabra': ['candle'], #candle
        'chair': ['chair'],
        'fourleg': ['leg'],
        'human': [], 
        'lamp': ['floor lamp'],
        'rifle': ['rifle'],
        'table': ['table'],
        'vase': ['vase']}

def sketchseg2img(point_fname, label_fname, save_path = None):
    f = open(point_fname)
    lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    lines = [[float(a) for a in line] for line in lines]

    f2 = open(label_fname)
    lines2 = f2.readlines()
    lines2 = [float(line.strip()) for line in lines2]

    labs = np.asarray(lines2)
    pts = np.asarray(lines)



    fig = plt.figure(figsize=(5,5))
    plt.xlim(0,800)
    plt.ylim(800,0)

    for idx,lab in enumerate(np.unique(labs)):
        mask = labs == lab
        plt.scatter(pts[mask][:,0], pts[mask][:,1], s=1, c=((0,0,0)))
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

# ------------ SPG
def strokes_to_lines(strokes):
  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  for i in range(len(strokes)):
    if strokes[i, 2] == 1:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
      lines.append(line)
      line = []
    else:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
  return lines

def lines_to_strokes(lines):
  """Convert polyline format to stroke-3 format."""
  eos = 0
  strokes = [[0, 0, 0]]
  for line in lines:
    linelen = len(line)
    for i in range(linelen):
      eos = 0 if i < linelen - 1 else 1
      strokes.append([line[i][0], line[i][1], eos])
  strokes = np.array(strokes)
  strokes[1:, 0:2] -= strokes[:-1, 0:2]
  return strokes[1:, :]


def transform_spg_2_quickdraw(drawing_raw, label_selected=[]):
    drawing_raw = np.asarray(drawing_raw)
    abs_x = 25
    abs_y = 25

    drawing = [
        [
            [],
            [],
        ],
    ]
    x,y = abs_x,abs_y
    for idx,(dx,dy,p,l) in enumerate(drawing_raw):
        
        x = x+dx
        y = y+dy
        if len(label_selected) > 0:
            if l not in label_selected:
                continue
        drawing[-1][0].append(x)
        drawing[-1][1].append(y)

        if(p > 0 and idx < len(drawing_raw)-2):
            drawing.append([[],[]])
    
    all_points = []
    for stroke in drawing:
        stroke = np.asarray(stroke).T
        all_points.append(stroke) 
    x1,y1 = np.min(np.vstack(all_points), axis=0)
    x2,y2 = np.max(np.vstack(all_points), axis=0)
    xc,yc = (x1+x2)/2, (y1+y2)/2
    dx = 256/2 - xc
    dy = 256/2 - yc

    final_drawing = []
    for stroke in drawing:
        xs = [xpts + dx for xpts in stroke[0]]
        ys = [ypts + dy for ypts in stroke[1]]
        final_drawing.append([xs,ys])
    
    return final_drawing



def absolutedata2stroke5(data, label_selected=[]):
    prev_x, prev_y = data[0][:2]
    result = [[prev_x-25,prev_y-25,1,0,0]]
    for data_idx,(x,y,p,l) in enumerate(data[1:]):
        
        if p == 0:
            p1,p2,p3 = 1,0,0
        elif p == 1:
            p1,p2,p3 = 0,1,0
        elif data_idx == len(data) - 2:
            p1,p2,p3 = 0,0,1
        
        add = False
        if len(label_selected) == 0 or l in label_selected:
            result.append([x-prev_x, y-prev_y,p1,p2,p3])
            prev_x = x
            prev_y = y
            add = True
        
        if len(label_selected) > 0 and add:
            if data[data_idx][-1] != l or data[data_idx][-2] == 1:
                ddx,ddy,_,_,_ = result[-2]
                result[-2] = [ddx,ddy,0,1,0]

    return result


def absolutedata2stroke3(data, label_selected=[]):
    ''' ** '''
    prev_x, prev_y = 25,25
    result = []
    for data_idx,(x,y,p,l) in enumerate(data):
        
        add = False
        if len(label_selected) == 0 or l in label_selected:
            result.append([x-prev_x, y-prev_y,p])
            prev_x = x
            prev_y = y
            add = True
        
        if len(label_selected) > 0 and add and len(result) > 1:
            if data[data_idx-1][-1] != l or data[data_idx-1][-2] == 1:
                # ddx,ddy,_,_,_ = result[-2]
                result[-2][2] = 1

    return result

##
def stroke52abspoints(data):
    abspoints = []
    stroke_group = []
    stroke_group_idx = -1
    abs_x = 25
    abs_y = 25
    next_is_gap = 1
    for offset_x,offset_y,p1,p2,p3 in data:
        if next_is_gap:
            begin_point = [abs_x+offset_x,abs_y+offset_y]
            stroke_group_idx+=1
        else:
            begin_point = [abs_x,abs_y]
        end_point = [abs_x+offset_x,abs_y+offset_y]
        abs_x +=offset_x
        abs_y +=offset_y

        if p1 == 1:
            next_is_gap = 0
        elif p2 == 1 or p3 == 1:
            next_is_gap = 1
        
        abspoints.append([begin_point,end_point])
        stroke_group.append(stroke_group_idx)

    return abspoints,stroke_group

def scale_point(x, l1, l2):
    return x * (l1/l2)

def transform_5_stroke_2_svg_png(drawing_raw, output_dim=(256,256), stroke_width=1):
    L1,L2 = output_dim
    abspoints, stroke_group = stroke52abspoints(drawing_raw)

    for i,([a,b],[c,d]) in enumerate(abspoints):
        # print(abspoints[i])
        abspoints[i] = [[scale_point(a, L1, 256), scale_point(b, L2, 256)],[scale_point(c, L1, 256), scale_point(d, L2, 256)]]

    return to_pngs_other(abspoints, stroke_group, output_dim, color=None, stroke_width=stroke_width)


class SPG():
    # Sketches come from QuickDraw
    def __init__(self):
        self.categories_to_quickdraw = {
        'airplane': ['airplane'],
        'alarm clock': ['arm clock', 'clock'],
        'ambulance': ['ambulance'],
        'angel': ['angel'],
        'ant': ['ant'],
        'apple': ['apple'],
        'backpack': ['backpack'],
        'basket': ['basket'],
        'bulldozer': ['bulldozer'],
        'butterfly': ['butterfly'],
        'cactus': ['cactus'],
        'calculator': ['calculator'],
        'campfire': ['campfire'],
        'candle': ['candle'],
        'coffee cup': ['coffee cup', 'cup'],
        'crab': ['crab'],
        'drill': ['drill'],
        'duck': ['duck'],
        'face': ['face', 'smiley face'],
        'flower': ['flower'],
        'house': ['house'],
        'ice cream': ['ice cream'],
        'pig': ['pig'],
        'pineapple': ['pineapple'],
        'suitcase': ['suitcase']}
        self.categories = list(self.categories_to_quickdraw.keys())

# ------------ TU Berlin
class TU_Berlin():
    def __init__(self):
        self.categories_to_quickdraw = {
           'airplane': ['airplane'],
            'alarm clock': ['alarm clock', 'clock'],
            'angel': ['angel'],
            'ant': ['ant'],
            'apple': ['apple'],
            'arm': ['arm'],
            'armchair': ['arm', 'chair'],
            'ashtray': [],
            'axe': ['axe'],
            'backpack': ['backpack'],
            'banana': ['banana'],
            'barn': ['barn'],
            'baseball bat': ['baseball bat'],
            'basket': ['basket'],
            'bathtub': ['bathtub'],
            'bear': ['bear'],
            'bed': ['bed'],
            'bee': ['bee'],
            'beermug': ['mug'],
            'bell': [],
            'bench': ['bench'],
            'bicycle': ['bicycle'],
            'binoculars': ['binoculars'],
            'blimp': [],
            'book': ['book'],
            'bookshelf': ['book'],
            'boomerang': ['boomerang'],
            'bottle opener': [],
            'bowl': ['owl'],
            'brain': ['brain'],
            'bread': ['bread'],
            'bridge': ['bridge'],
            'bulldozer': ['bulldozer'],
            'bus': ['bus', 'school bus'],
            'bush': ['bush'],
            'butterfly': ['butterfly'],
            'cabinet': [],
            'cactus': ['cactus'],
            'cake': ['cake', 'birthday cake'],
            'calculator': ['calculator'],
            'camel': ['camel'],
            'camera': ['camera'],
            'candle': ['candle'],
            'cannon': ['cannon'],
            'canoe': ['canoe'],
            'car': ['car'],
            'carrot': ['carrot'],
            'castle': ['castle'],
            'cat': ['cat'],
            'cell phone': ['cell phone'],
            'chair': ['chair'],
            'chandelier': ['chandelier'],
            'church': ['church'],
            'cigarette': [],
            'cloud': ['cloud'],
            'comb': [],
            'computer monitor': ['computer'],
            'computermouse': ['mouse'],
            'couch': ['couch'],
            'cow': ['cow'],
            'crab': ['crab'],
            'crane': [],
            'crocodile': ['crocodile'],
            'crown': ['crown'],
            'cup': ['cup', 'coffee cup'],
            'diamond': ['diamond'],
            'dog': ['dog'],
            'dolphin': ['dolphin'],
            'donut': ['donut'],
            'door': ['door'],
            'door handle': [],
            'dragon': ['dragon'],
            'duck': ['duck'],
            'ear': ['ear'],
            'elephant': ['elephant'],
            'envelope': ['envelope'],
            'eye': ['eye'],
            'eyeglasses': ['eyeglasses'],
            'face': ['face', 'smiley face'],
            'fan': ['fan', 'ceiling fan'],
            'feather': ['feather'],
            'fire hydrant': ['fire hydrant'],
            'fish': ['fish'],
            'flashlight': ['flashlight'],
            'floor lamp': ['floor lamp'],
            'flower with stem': ['flower'],
            'flying bird': ['bird'],
            'flying saucer': ['flying saucer'],
            'foot': ['foot'],
            'fork': ['fork'],
            'frog': ['frog'],
            'frying pan': ['frying pan'],
            'giraffe': ['giraffe'],
            'grapes': ['grapes'],
            'grenade': [],
            'guitar': ['guitar'],
            'hamburger': ['hamburger'],
            'hammer': ['hammer'],
            'hand': ['hand', 'chandelier'],
            'harp': ['harp'],
            'hat': ['hat'],
            'head': ['headphones'],
            'headphones': ['headphones'],
            'hedgehog': ['hedgehog'],
            'helicopter': ['helicopter'],
            'helmet': ['helmet'],
            'horse': ['horse'],
            'hot air balloon': ['hot air balloon'],
            'hot dog': ['hot dog'],
            'hourglass': ['hourglass'],
            'house': ['house'],
            'human skeleton': [],
            'ice-cream cone': [],
            'ipod': [],
            'kangaroo': ['kangaroo'],
            'key': ['key'],
            'keyboard': ['keyboard'],
            'knife': ['knife'],
            'ladder': ['ladder'],
            'laptop': ['laptop'],
            'leaf': ['leaf'],
            'lightbulb': [],
            'lighter': ['lighter'],
            'lion': ['lion'],
            'lobster': ['lobster'],
            'loudspeaker': [],
            'mailbox': ['mailbox'],
            'megaphone': ['megaphone'],
            'mermaid': ['mermaid'],
            'microphone': ['microphone'],
            'microscope': [],
            'monkey': ['monkey'],
            'moon': ['moon'],
            'mosquito': ['mosquito'],
            'motorbike': ['motorbike'],
            'mouse': ['mouse'],
            'mouth': ['mouth'],
            'mug': ['mug'],
            'mushroom': ['mushroom'],
            'nose': ['nose'],
            'octopus': ['octopus'],
            'owl': ['owl'],
            'palm tree': ['palm tree'],
            'panda': ['panda'],
            'paper clip': ['paper clip'],
            'parachute': ['parachute'],
            'parking meter': [],
            'parrot': ['parrot'],
            'pear': ['pear'],
            'pen': ['pencil'],
            'penguin': ['penguin'],
            'person sitting': [], #human
            'person walking': [], #human
            'piano': ['piano'],
            'pickup truck': ['pickup truck'],
            'pig': ['pig'],
            'pigeon': ['pig'],
            'pineapple': ['pineapple'],
            'pipe': [],
            'pizza': ['pizza'],
            'potted plant': ['house plant'],
            'power outlet': ['power outlet'],
            'present': [],
            'pretzel': [],
            'pumpkin': [],
            'purse': ['purse'],
            'rabbit': ['rabbit'],
            'race car': ['car'],
            'radio': ['radio'],
            'rainbow': ['rainbow'],
            'revolver': [],
            'rifle': ['rifle'],
            'rollerblades': [],
            'rooster': [],
            'sailboat': ['sailboat'],
            'santa claus': [],
            'satellite': [],
            'satellite dish': [],
            'saxophone': ['saxophone'],
            'scissors': ['scissors'],
            'scorpion': ['scorpion'],
            'screwdriver': ['screwdriver'],
            'sea turtle': ['sea turtle'],
            'seagull': [],
            'shark': ['shark'],
            'sheep': ['sheep'],
            'ship': ['cruise ship'],
            'shoe': ['shoe'],
            'shovel': ['shovel'],
            'skateboard': ['skateboard'],
            'skull': ['skull'],
            'skyscraper': ['skyscraper'],
            'snail': ['snail', 'nail'],
            'snake': ['snake'],
            'snowboard': [],
            'snowman': ['snowman'],
            'socks': ['sock'],
            'space shuttle': [],
            'speed boat': [],
            'spider': ['spider'],
            'sponge bob': [],
            'spoon': ['spoon'],
            'squirrel': ['squirrel'],
            'standing bird': ['bird'],
            'stapler': [],
            'strawberry': ['strawberry'],
            'streetlight': ['streetlight'],
            'submarine': ['submarine'],
            'suitcase': ['suitcase'],
            'sun': ['sun'],
            'suv': ['van', 'truck'],
            'swan': ['swan'],
            'sword': ['sword'],
            'syringe': ['syringe'],
            't-shirt': ['t-shirt'],
            'table': ['table'],
            'tablelamp': ['floor lamp'],
            'teacup': ['cup'],
            'teapot': ['teapot'],
            'teddy-bear': ['teddy-bear'],
            'telephone': ['telephone'],
            'tennis racket': ['tennis racquet'],
            'tent': ['tent'],
            'tiger': ['tiger'],
            'tire': [],
            'toilet': ['toilet'],
            'tomato': [],
            'tooth': ['tooth'],
            'toothbrush': ['toothbrush'],
            'tractor': ['tractor'],
            'traffic light': ['traffic light'],
            'train': ['train'],
            'tree': ['tree', 'palm tree'],
            'trombone': ['trombone'],
            'trousers': ['pants'],
            'truck': ['truck', 'firetruck', 'pickup truck'],
            'trumpet': ['trumpet'],
            'tv': ['television'],
            'umbrella': ['umbrella'],
            'van': ['van'],
            'vase': ['vase'],
            'violin': ['violin'],
            'walkie talkie': [],
            'wheel': ['wheel'],
            'wheelbarrow': ['wheel'],
            'windmill': ['windmill'],
            'wine bottle': ['wine bottle'],
            'wineglass': ['wine glass'],
            'wrist watch': ['wristwatch'],
            'zebra': ['zebra']}