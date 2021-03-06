# conda activate darkflow

import os
import matplotlib.pyplot as plt 
import cv2 
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from generate_xml import write_xml
import sys

# global constants
img = None 
tl_list = []
br_list = [] 
object_list  = [] 
print(sys.argv)
# constants 

#image_folder = 'C:\\Users\\saara\\projects\\darkflow_yolov2_custom_model\\training_data\\oneclassimages\\Person'
image_folder = 'images'
savedir = 'annotations'
obj = 'scissors'

# create a callback function
# first click is top left corner
def line_select_callback(click, release):
    global tl_list
    global br_list
    tl_list.append((int(click.xdata), int(click.ydata)))
    br_list.append((int(release.xdata), int(release.ydata)))
    object_list.append(obj)

def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
       
    if event.key == 'q':
        write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        tl_list = []
        br_list = []
        object_list = []
        img = None
        #plt.close()

def toggle_selector(event):
    toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1) 
        
        # # where the image is displayed on the screen
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setPosition(250, 120, 1280, 1024)
        #
        image = cv2.imread(image_file.path)
        # matplotlib and opencv work in different color spaces
        # we need to convert the colors, cv2 = BGR, matplotlib = RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype ='box', useblit=True,
            # 1 means left mouse click, values from matplotlib documentation
            button=[1], minspanx =5, minspany=5,
            spancoords='pixels', interactive=True
        )

        # Connect callbacks
        # send mouse button event to callback function
        bbox = plt.connect('button_press_event', toggle_selector)
        key = plt.connect('key_press_event', onkeypress)
        plt.tight_layout()
        plt.show()
        plt.close()
