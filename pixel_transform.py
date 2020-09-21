from tkinter import Tk, filedialog, Frame, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import filetype


# set up an a list for files
images = []
# to store the image file types
filetypes = []


# Load the image referred to by path
def opencv_img(path):
    # read and convert image
    image = cv2.imread(path)
    cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    return(image)

# Convert it to ImageTK
# necessary to use cvtColor to correct to expected RGB color
def convert_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    return(imgtk)

# Wrapper to load the image for display
def load_img(count):
    return convert_img(opencv_img(count))

#Exit the program
def quit_img(event):
    root.destroy() #Kill the display
    sys.exit(0)

# Select the image to load
def select_image(event):
    # Prompt the user
    path = filedialog.askopenfilename()
    # if there is a path and it is readable
    if len(path) > 0 and cv2.haveImageReader(path):
        update_original(path)
    else:
        print("no image")
  
# Negative image
def neg_img(event):
    global image
    neg_img = 255-image
    print("HI")
    update_new(neg_img)
    
  
def update_original(path):
    global original, image
    image = opencv_img(path)
    disp_img = convert_img(image)
    
    original.configure(image=disp_img)
    original.image = disp_img
    return disp_img

def update_new(img):    
    global new
    disp_img = convert_img(img)
    new.configure(image=disp_img)
    new.image = disp_img

    

    

def main():

    # Root window
    global root, original, new
    root = Tk()
    # The original loaded image
    original = Label(image=None)
    original.pack(side="top", padx=10, pady=10)
    
    # The new modifed image
    new = Label(image=None)
    new.pack(side="top", padx=10, pady=10)
    
    # Frame for buttons
    frame = Frame()
    frame.pack()
    
        # button for select image
    btn_neg = Button(
        master = frame,
        text = "Negative",
        underline = 0
    )
    btn_neg.grid(row = 0, column = 1)
    btn_neg.bind('<ButtonRelease-1>', neg_img)
    
    
    # button for select image
    btn_select = Button(
        master = frame,
        text = "Select an Image",
        underline = 0
    )
    btn_select.grid(row = 1, column = 1)
    btn_select.bind('<ButtonRelease-1>', select_image)
    

    # Bind all the required keys to functions

    root.bind("<q>", quit_img)
    
    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()
