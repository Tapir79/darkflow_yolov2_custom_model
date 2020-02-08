# conda activate darkflow

import os
import matplotlib.pyplot as plt 
import cv2 
from matplotlib.widgets import RectangleSelector
from pathlib import Path

# global constants
img = None 
tl_list = []
br_list = [] 
object_list  = [] 

# constants 

image_folder = 'C:\\Users\\saara\\projects\\darkflow_yolov2_custom_model\\training_data\\oneclassimages\\Person'
savedir = 'annotations'
obj = 'person'

# create a callback function
# first click is top left corner
def line_select_callback(click, release):
    global tl_list
    global br_list
    tl_list.append((int(click.xdata), int(click.ydata)))
    br_list.append((int(release.xdata), int(release.ydata)))

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1) 
        # where the image is displayed on the screen
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(250, 120, 1280, 1024)
        #
        image = cv2.imread(image_file.path)
        # matplotlib and opencv work in different color spaces
        # we need to convert the colors, cv2 = BGR, matplotlib = RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype ='box', useblit=True,
            button=[1], minspanx =5, minspany=5,
            spancoords='pixels', interactive=True
        )
        # send mouse button event to callback function
        plt.connect('button_press_event', line_select_callback)
        plt.show()
