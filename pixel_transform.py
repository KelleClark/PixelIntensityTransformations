from tkinter import Tk, filedialog, Frame, Label, Button
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import filetype
import math


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
    update_new(neg_img)

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
    
    
def prompt_bitplane(event):
    colors = ["blue", "green", "red"]
    while(True):
        color = simpledialog.askstring("Input", "What color? (red, green, or blue)",
                                       parent=root)
        if color != None and color.lower() in colors:
            color_code = colors.index(color.lower())
            break

    while (True):
        bit = simpledialog.askinteger("Input", "What bit value? (0-7)",
                                         parent=root,
                                         minvalue=0, maxvalue=7)
        if bit != None:
            break
        
    bitplane(color_code, bit)


def log_trans(event):
    global image
    
    log_img = image.copy()
    # Prevent overflow
    log_img[log_img<255] += 1
    
    log = np.log(log_img)
    
    #Log transformation 
    log_img = (255/np.max(log)) * log
    log_img = np.array(log_img, dtype = np.uint8)
    
    update_new(log_img)
    
  
def update_original(path):
    global original, image
    image = opencv_img(path)
    disp_img = convert_img(image)
    
    original.configure(image=disp_img)
    original.image = disp_img
    return disp_img

def update_new(img):    
    global new, new_img
    new_img = img
    disp_img = convert_img(img)
    new.configure(image=disp_img)
    new.image = disp_img
    
def save(event):
    name = filedialog.asksaveasfilename(confirmoverwrite=True)
    cv2.imwrite(name, new_img)
    
    

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
    
    # button for negative image
    btn_neg = Button(
        master = frame,
        text = "Negative",
        underline = 0
    )
    btn_neg.grid(row = 0, column = 0)
    btn_neg.bind('<ButtonRelease-1>', neg_img)
    
    # button for bitpiece
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

    
    # button for select image
    btn_select = Button(
        master = frame,
        text = "Select an Image",
        underline = 0
    )
    btn_select.grid(row = 1, column = 1)
    btn_select.bind('<ButtonRelease-1>', select_image)
    
    # button for select image
    btn_save = Button(
        master = frame,
        text = "Save Image",
        underline = 0
    )
    btn_save.grid(row = 1, column = 0)
    btn_save.bind('<ButtonRelease-1>', save)
    
    

    # Bind all the required keys to functions

    root.bind("<q>", quit_img)
    
    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()
