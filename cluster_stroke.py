import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import PIL
import read_datasets as rd
import constants as CONST
import pickle

import sys
sys.path.insert(0, "/home/xiaoyuz1/mnist-em-bmm-gmm")
import gmm
import kmeans

'''
Should be very similar to what the jupyter notebook stroke_clustering is doing.
'''

def render_stroke_data(strokes):
    '''
    strokes: numpy array of shape (num_strokes, num_points, 2)
    '''