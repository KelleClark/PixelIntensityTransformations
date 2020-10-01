# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image, ImageTk
from time import time
from tkinter import *
from tkinter.messagebox import showinfo
import argparse
import cv2
import filetype
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# set up an a list for files
images = []

# to store the image file types
filetypes = []

# dimensions of each image, stored as strings
columns = []
rows = []
pixels = []

# to store total pixels and intensity, k, for image, stored as int
intensity = []
bitsize = []

# index for the list of images in the browser
count = 0

# Current manipulation, used to save images
manipul = ""

##-------------Functions for Instantiating the Program-----------------------##
# Get and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image browser v1.0')
    parser.add_argument('path', metavar='dir',
                        help='The root directory to view photos in')
    parser.add_argument('--rows', type=int,  default=720,
                        help='Max number of rows on screen  (Default is 720)')
    parser.add_argument('--cols',  type=int, default=1080,
                        help='Max number of columns on screen (Default is 1080)')

    args = parser.parse_args()
    return(args)

# Check for images in the path and save the exact path to each
#   image in a list.
def load_path(path):
    global images
    rootDir = os.path.join(path)
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pos_img = dirName + "/" + fname
            if cv2.haveImageReader(pos_img): # if it is a readable image
                images.append(pos_img)  #add it to the list of images
                filetypes.append(filetype.guess(pos_img).mime) #Get the file type and save it
                # add placeholders for image column count, row count, and pixel count to lists
                columns.append("")
                rows.append("")
                pixels.append("")
                bitsize.append(int(os.path.getsize(pos_img))*8) #at first this is the size of the image in bits
                intensity.append(0) #constructed to be 0 until opencv_img finds the intensity

    # If there is a problem with the given path, exit
    if len(images) == 0:
        print("Invalid path or there are no images in path")
        sys.exit(1)
##---------------------------------------------------------------------------##

##-------Functions to open/read an image file and rendering in UI------------##

# Read in image and conform to fit window
def opencv_img(count):
    # read and convert image
    image = cv2.imread(images[count])
    columns[count] = str(image.shape[1]) # add column count to list
    rows[count] = str(image.shape[0]) # add row count to list
    pixels[count] = str(image.shape[1] * image.shape[0]) # add pixel count to list
    intensity[count] = math.ceil(int(bitsize[count])/(3*int(image.shape[0])*int(image.shape[1])))

    # Set scale multiplier to the lowest of the following values:
    # 1
    # window row count / image row count
    # window column count / image column count
    scale = min(1, min(get_args().rows / image.shape[0], get_args().cols / image.shape[1]))

    # Set triangle corners used for affine transformation to top left, top right, and bottom left corners of image
    srcTri = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1]]).astype(np.float32)

    # Set location of top right and bottom left corners of resized image
    dstTri = np.array( [[0, 0], [int(image.shape[1] * scale), 0], [0, int(image.shape[0] * scale)]] ).astype(np.float32)

    # Perform affine transformation to resize image
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    image = cv2.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))

    # Trim black border from resized image
    image = image[0:int(image.shape[0] * scale), 0:int(image.shape[1] * scale)]
    return(image)

# Convert it to ImageTK
# necessary to use cvtColor taking from BGR to expected RGB color
def convert_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    return(imgtk)

# Wrapper to load the image for display
def load_img(count):
    return convert_img(opencv_img(count))
##---------------------------------------------------------------------------##

##--------Functions to display the metadata of images------------------------##
# Gets filename and file location
def extract_meta():
    global count
    ind = images[count].rindex("/")
    ans = ['','']
    if ind != -1:
        ans[0] = images[count][0:ind]
        ans[1]= images[count][ind+1:]

    return ans

# Show metadata
def meta(event):
    global count
    impath = images[count]
    info = os.lstat(impath)
    showinfo("Image Metadata", info)

# Update the information about the newest photo and the image itself
#   on the window
def update_window(imgtk, tex):
        label['image'] = imgtk
        label['text'] = tex[1]+"\nfrom "+tex[0]+"\n"+columns[count]+" x "+ \
        rows[count]+" ("+pixels[count]+" pixels)\nImage type: "+ \
        filetypes[count]+ "\nFile size: "+str(os.lstat(images[count]).st_size)\
        +" bytes\n with intensity "+str(intensity[count])
        label.photo = imgtk

##----------User Controls for the UI-----------------------------------------##
# Go to next image
def next_img(event):
    global count
    if count >= len(images) -1:
        count = -1 # -1 to offset regular function
    count = count + 1  # Next image in the list
    imgtk = load_img(count)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

# Go to prior image
def prev_img(event):
    global count
    if count <= 0:
        count = (len(images) - 1) + 1 # +1 to offset regular function
    count = count - 1  # Prior image in the list
    imgtk = load_img(count)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)
    
    #Exit the program
def quit_img(event):
    root.destroy() #Kill the display
    sys.exit(0)
    
# save the image to the main given path appending the name of any transformation
def save(event):
    global count
    cv2.imwrite(images[count].rstrip(".jpg")+manipul+".jpg", img)
    

##---------------------------------------------------------------------------##

def main():

    #Get the command arguments
    args = get_args()

    # Root window
    global root
    root = Tk()
    load_path(args.path)
    imgtk = load_img(count)
    tex = extract_meta()

    # Put everything in the display window
    global label
    label = Label(root, text = tex[1]+"\nfrom "+tex[0]+"\n"+columns[count]+ \
    " x " +rows[count]+" ("+pixels[count]+" pixels)\nImage type: "+ \
    filetypes[count]+"\nFile size: "+str(os.lstat(images[count]).st_size)+ \
    " bytes", compound = RIGHT, image=imgtk)
    label.pack()

    # Frame to display navigation buttons at bottom of window
    frame = Frame()
    frame.pack()

    # Button for prior image
    btn_previous = Button(
        master = frame,
        text = "Previous",
        underline = 0
    )
    btn_previous.grid(row = 0, column = 0)
    btn_previous.bind('<ButtonRelease-1>', prev_img)
    
    # Button for next image
    btn_next = Button(
        master = frame,
        text = "Next",
        underline = 0
    )
    btn_next.grid(row = 0, column = 1)
    btn_next.bind('<ButtonRelease-1>', prev_img)
    
     
    # Button for next image
    btn_save = Button(
        master = frame,
        text = "Save",
        underline = 0
    )
    btn_save.grid(row = 0, column = 2)
    btn_save.bind('<ButtonRelease-1>', save_img)
    
      # Bind all the required keys to functions
    root.bind('<n>', next_img)
    root.bind("<p>", prev_img)
    root.bind("<q>", quit_img)
    root.bind("<s>", save_img)
   

    root.mainloop() # Start the GUI
    
if __name__ == "__main__":
    main()
    

