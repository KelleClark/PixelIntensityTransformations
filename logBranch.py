# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tkinter import Tk, filedialog, Frame, Label, Button, simpledialog, filedialog
from PIL import Image, ImageTk
from time import time
from tkinter.messagebox import showinfo
import argparse
import cv2
import filetype
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys



##-------Functions to open/read an image file and rendering in UI------------##

# Read in image referred to by path and conform to fit screen
def opencv_img(path):
    # read and convert image
    image = cv2.imread(path)
    img = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    return(img)
    
    
# Convert it to ImageTK
# necessary to use cvtColor taking from BGR to expected RGB color
def convert_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    return(imgtk)
##---------------------------------------------------------------------------##

##--------Functions to display the metadata of images------------------------##

##----------User Controls for the UI-----------------------------------------##

# Select the image to load
def select_img(event):
    # Prompt the user
    path = filedialog.askopenfilename()
    # if there is a path and it is readable
    if len(path) > 0 and cv2.haveImageReader(path):
        update_original(path)
    else:
        print("no image")

#Exit the program
def quit_img(event):
    root.destroy() #Kill the display
    sys.exit(0)
    
# Save the image to the main given path appending the name of any transformation
def save_img(event):
    global new
    name = filedialog.asksaveasfilename(confirmoverwrite=True)
    cv2.imwrite(name, new)
    

##---------------------------------------------------------------------------##

##---------GUI update image formating ---------------------------------------##
# User given path to image, open and format image return disp_img        
def update_original(path):
    global original, image
    image = opencv_img(path)
    disp_img = convert_img(image) 
    original.configure(image=disp_img)
    original.image = disp_img
    return disp_img
   

# A newly transformed image, new, is formatted for display
def update_new(img):      
    global new, new_img
    new_img = img
    disp_img = convert_img(img)
    new.configure(image=disp_img)
    new.image = disp_img
    
   
##---------------------------------------------------------------------------##
##---------Pixel Transformations---------------------------------------------##

   
# Negative image
def neg_img(event):
    global image
    neg_img = 255-image
    print("HI")
    update_new(neg_img)

# Collect user chosen parameters for bitplane transformation
def prompt_bitplane(event):
    colors = ["blue", "green", "red"]
    while(True):
        color = simpledialog.askstring("Input", "What color? (red, green, or blue)",
                                       parent=root)
        if color.lower() in colors:
            color_code = colors.index(color)
            break

    while (True):
        bit = simpledialog.askinteger("Input", "What bit value? (0-7)",
                                         parent=root,
                                         minvalue=0, maxvalue=7)
        if bit != None:
            break
        
    bitplane(color_code, bit)
 
def bitplane(color, bit):
    global image
    
    # Faster numpy trick
    bitplane_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    bitplane_img[:,:,color][image[:,:,color]% 2**(bit+1) >= 2**bit] = np.uint8(255)
    
    """for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][color] % 2**(bit+1) >= 2**bit:
                bitplane_img[i][j][color] = 255"""
                
    update_new(bitplane_img)
    
def log_trans(event):
    global image
    
    log_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    cmax_red = np.amax(image[:,:,0])
    cmax_green = np.amax(image[: ,:, 1])
    cmax_blue = np.amax(image[: ,:,2])
    
    if cmax_red == 255:
        cred = 255/math.log(255,10)
    elif cmax_red < 10:
        cred = 1
    else:
        cred = 255/math.log(cmax_red + 1,10)
        
    if cmax_green == 255:
        cgreen = 255/math.log(.01,10)
    elif cmax_green < 10:
        cgreen = 1
    else:
        cgreen = 255/math.log(cmax_green + 1,10) 
        
    if cmax_blue == 255:
        cblue = 255/math.log(.01,10)
    elif cmax_blue < 10:
        cblue = 1
    else:
        cblue = 255/math.log(cmax_blue + 1 ,10)
    
    for i in range(0, image.shape[0]-1):
        for j in range(0, image.shape[1]-1):
            log_img[i,j,0] = np.uint8(255 * math.log(1 + image[i,j,0],10)/math.log(1+ cmax_red,10))
            log_img[i,j,1] = np.uint8(cgreen * math.log(1 + image[i,j,1],10))
            log_img[i,j,2] = np.uint8(cblue * math.log(1 + image[i,j,2],10))

   
    update_new(log_img)
        
    
##---------------------------------------------------------------------------##
def main():
    global root, label, original, new

   
    root = Tk()
    
    # The original loaded image
    original = Label(image=None)
    original.pack(side="top", padx=10, pady=10)
    
    # The new modifed image
    new = Label(image=None)
    new.pack(side="top", padx=10, pady=10)

    # Frame to display navigation buttons at bottom of window
    frame = Frame()
    frame.pack()
 
     
    # Button for save_img image
    btn_save = Button(
        master = frame,
        text = "Save",
        underline = 0
    )
    btn_save.grid(row = 0, column = 2)
    btn_save.bind('<ButtonRelease-1>', save_img)
    
    # Button for select image
    btn_select = Button(
        master = frame,
        text = "Select an Image",
        underline = 0
    )
    btn_select.grid(row = 4, column = 0)
    btn_select.bind('<ButtonRelease-1>', select_img)
    
    # Button for negative of image
    btn_neg = Button(
        master = frame,
        text = "Negative",
        underline = 0
    )
    btn_neg.grid(row = 0, column = 0)
    btn_neg.bind('<ButtonRelease-1>', neg_img)
    
    
    # button for bitplane
    btn_bit = Button(
        master = frame,
        text = "Bitplane",
        underline = 0
    )
    btn_bit.grid(row = 0, column = 1)
    btn_bit.bind('<ButtonRelease-1>', prompt_bitplane)
    
    # Button for log transformation of image
    btn_log = Button(
        master = frame,
        text = "Log",
        underline = 0
    )
    btn_log.grid(row = 0, column = 2)
    btn_log.bind('<ButtonRelease-1>', log_trans)
    
    
    # Bind all the required keys to functions
    root.bind("<q>", quit_img)
    root.bind("<s>", save_img)
   

    root.mainloop() # Start the GUI
    
if __name__ == "__main__":
    main()
    

