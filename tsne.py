import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)
import cv2 

x = np.load('/Users/erinz/Documents/face/face_x.npy') * 2000
y = np.load('/Users/erinz/Documents/face/face_y.npy') * 2000
arr = []
image_paths = np.load('/Users/erinz/Documents/face/face_image_path.npy')
for path in image_paths:
    image_path = path.replace("/raid/xiaoyuz1/sketch_datasets/spg/face", "/Users/erinz/Documents/face")
    image = cv2.imread(image_path)
    arr.append(image) #[:,:,0]
arr = np.asarray(arr)

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x,y, ls="", marker="o")
im = OffsetImage(arr[0], zoom=0.4)
xybox=(1.05,1.05)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords=("axes fraction", "data"),  pad=0.3,  arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        print(line.contains(event)[1])
        ind = line.contains(event)[1]["ind"][0]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ## ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        ab.xybox = (1.05, 1.05)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(arr[ind])
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)           
plt.show()
