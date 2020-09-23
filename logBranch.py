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

# Read in image and conform to fit window
def opencv_img(path):
    # read and convert image
    image = cv2.imread(path)
    img = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 
    return(img)
    
    
# Convert it to ImageTK
# necessary to use cvtColor taking from BGR to expected RGB color
def convert_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    return(imgtk)

# Wrapper to load the image for display
#def load_img(count):
#    return convert_img(opencv_img(count))
##---------------------------------------------------------------------------##

##--------Functions to display the metadata of images------------------------##
# Gets filename and file location
def extract_meta():
    ind = images[count].rindex("/")
    ans = ['','']
    if ind != -1:
        ans[0] = images[count][0:ind]
        ans[1]= images[count][ind+1:]

    return ans

# Show metadata
def meta(event):
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
    name = filedialog.asksaveasfilename(confirmoverwrite=True)
    cv2.imwrite(name, new_img)
    

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
    global new
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
    
    max_red = np.amax(image[:,:,0])
    max_green = np.amax(image[:,:,1])
    max_blue = np.amax(image[:,:,2])
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            log_img[i, j, 0] = 255/(math.log(2,1+max_red))*math.log(1+image[i,j,0])
            log_img[i, j, 1] = 255/(math.log(2,1+max_green))*math.log(1+image[i,j,1])
            log_img[i, j, 2] = 255/(math.log(2,1+max_blue))*math.log(1+image[i,j,2])
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
    

