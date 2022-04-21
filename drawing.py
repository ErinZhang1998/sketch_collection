import csv
import seaborn as sns
import numpy as np
import pandas as pd

import json
import matplotlib.pyplot as plt

def create_word_list(sentences):
    res = []
    for sentence in sentences:
        l = sentence.split()
        if '' in l:
            l.remove('')
        res += l
    return res
    
def get_xy(answer_json):
    
    try:
        answer_dict = json.loads(answer_json)['data']
    except:
        raise
    
    actions = answer_dict['actions']
    text = answer_dict['actionsName']
    action_id = answer_dict['id']
    
    xy = []
    for d in actions:
        if d['type'] == 0:
            xy.append([])
        xy[-1] += [[float(d['x']), float(d['y'])]]
    return xy, text, action_id


class Drawing:
    def __init__(self, df):
        self.df = df
        
        self.num_workers = len(df['WorkerId'].unique())
        data = {}
        total = 0
        
        for i in range(len(df)):
            
            df_row = df.iloc[i]
            this_worker_drawings = {}

            for k,v in df_row.items():        
                if not k.startswith("Answer.canvasJsonOneStepName"):
                    continue
                if v is np.nan:
                    continue

                canvas_id = int(k.split("__")[0].split("_")[1])
                if canvas_id == 0:
                    continue
                prompt = df_row["Input.PROMPT0{}".format(canvas_id)]

                try:
                    part_xy, text, action_id = get_xy(v)
                    text = text.lower()

                    if canvas_id not in this_worker_drawings:
                        
                        this_worker_drawings[canvas_id] = {
                            'df_index' : i,
                            'canvas_id' : canvas_id,
                            'prompt' : prompt,
                            'data' : [],
                            'WorkerId' : df_row['WorkerId'],
                            'time' : df_row['WorkTimeInSeconds'],
                        }
                    
                    this_worker_drawings[canvas_id]['data'].append({
                        'annotation' : create_word_list([text.lower()]),
                        'drawing' : part_xy,
                    })
                except:
                    continue
            for k,v in this_worker_drawings.items():

                # if "sleepy" in v['prompt']:
                #     print(v['prompt'], total)
                
                data[total] = v
                total += 1
        
        self.data = data
    
    def all_drawings(self):
        drawing_df = []
        for i,datai in self.data.items():
            drawingi = []
            for dataij in datai['data']:
                part_xy = dataij['drawing']
                
                if(len(part_xy) > 1):
                    part_xy = np.vstack(part_xy)
                    # part_xy = np.expand_dims(part_xy, axis=0)
                else:
                    part_xy = np.asarray(part_xy)
                    part_xy = np.squeeze(part_xy, axis=0)
                    
                drawingi += [part_xy]
            drawing_df.append(drawingi)
        return drawing_df
    
    def draw_all(self):
        hue_palette = sns.color_palette("husl", 20)
        num_plots_per_row = 5
        num_rows = int(np.ceil(len(self.data) / num_plots_per_row))
        plt.figure(figsize=(num_plots_per_row * 3, num_rows * 3,))
        
        plot_idx_acc = 1
        
        for data_idx in range(len(self.data)):
            data_idx = int(data_idx)
            datai = self.data[data_idx]['data']
            prompt = self.data[data_idx]['prompt']

            ax = plt.subplot(num_rows, num_plots_per_row , plot_idx_acc)
            plt.xlim(0,400)
            plt.ylim(400,0)
            text = None
            components_accumatlor = []
            for dataij in datai:
                components_accumatlor += dataij['drawing']
            
            for part_xy_i in components_accumatlor:
                part_xy_i = np.asarray(part_xy_i)
                plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0)) #hue_palette[plot_idx_acc]
            
            ax.title.set_text(
                "{}, {}".format(data_idx, prompt),
            )
            ax.title.set_fontsize(9)
            # plt.annotate(text, xy=(0,0), backgroundcolor='w', fontsize=9)
            plot_idx_acc += 1        
        
        plt.show()


    def find(self, prompt):
        for data_idx in range(len(self.data)):
            data_idx = int(data_idx)
            if prompt in self.data[data_idx]['prompt']:
                self.draw(data_idx)
    
    def find_in_text(self, text_part):
        for data_idx in range(len(self.data)):
            data_idx = int(data_idx)
            skip = self.draw(data_idx, text_part)
    
    def draw_per_stroke(self, data_idx):
        data_idx = self.data[data_idx]
        datai = data_idx['data']
        prompt = data_idx['prompt']

        hue_palette = sns.color_palette("husl", 20)
        total_steps = sum([len(l['drawing']) for l in datai])
        
        num_plots_per_row = 3
        num_rows = int(np.ceil(total_steps / num_plots_per_row))
        plot_idx_acc = 1
        plt.figure(figsize=(num_plots_per_row * 5, num_rows * 5,))
        
        components_accumatlor = []
        for dataij in datai:
            text = dataij['annotation']
            part_xy = dataij['drawing']
            
            for part_xy_idx, stroke_data in enumerate(part_xy):
                plt.subplot(num_rows, num_plots_per_row , plot_idx_acc)
                plt.xlim(0,400)
                plt.ylim(400,0)

                components_accumatlor += [stroke_data]
                
                for stroke_idx, part_xy_i in enumerate(components_accumatlor):
                    part_xy_i = np.asarray(part_xy_i)
                    
                    if stroke_idx == len(components_accumatlor) - 1:
                        plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0)) #hue_palette[plot_idx_acc]
                    else:
                        plt.plot(part_xy_i[:,0], part_xy_i[:,1], c='grey') #hue_palette[plot_idx_acc]
                # if part_xy_idx == 0:
                #     plt.annotate(" ".join(text), xy=(0,0), backgroundcolor='w', fontsize=9)
                plot_idx_acc += 1
            
        
        # plt.title("{} {}".format(data_idx, prompt))
        plt.show()

    
    def draw(self, data_idx, text_part = None):
        print("Index in dataframe: ", self.data[data_idx]['df_index'])
        print("Canvas ID: ", self.data[data_idx]['canvas_id'])
        

        datai = self.data[data_idx]['data']
        prompt = self.data[data_idx]['prompt']
        print(data_idx, prompt)

        hue_palette = sns.color_palette("husl", 20)
        total_steps = len(datai)
        
        num_plots_per_row = 3
        num_rows = int(np.ceil(total_steps / num_plots_per_row))
        plot_idx_acc = 1
        plt.figure(figsize=(num_plots_per_row * 10, num_rows * 10,))
        
        all_texts = []
        components_accumatlor = []
        if text_part is not None:
            skip_whole = True
            for dataij in datai:
                
                text = dataij['annotation']
                if text_part in text:
                    skip_whole = False

            if skip_whole:
                return

        for dataij in datai:
            
            text = dataij['annotation']
            all_texts.append(" ".join(text))
            # if text_part is not None:
            #     if text_part not in text:
                    
            #         continue
            
            ax = plt.subplot(num_rows, num_plots_per_row , plot_idx_acc)
            plt.xlim(0,400)
            plt.ylim(400,0)

            part_xy = dataij['drawing']

            components_accumatlor += part_xy
            for part_xy_i in components_accumatlor:
                part_xy_i = np.asarray(part_xy_i)
                plt.plot(part_xy_i[:,0], part_xy_i[:,1], c=(0,0,0)) #hue_palette[plot_idx_acc]

            # plt.annotate(text, xy=(0,0), backgroundcolor='w', fontsize=40)
            plot_idx_acc += 1
            ax.axis('off')
        
        # plt.title("{} {}".format(data_idx, prompt))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        for t in all_texts:
            print(t)
        # print(all_texts)


