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

def bitplane(color, bit):
    global image
    bitplane_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][color] % 2**(bit+1) < 2**bit:
                bitplane_img[i][j][color] = 0
            else:
                bitplane_img[i][j][color] = 255

    update_new(bitplane_img)

def bitplane_blue_0(event):
    bitplane(0,0)
    
def bitplane_blue_1(event):
    bitplane(0,1)
    
def bitplane_blue_2(event):
    bitplane(0,2)

def bitplane_blue_3(event):
    bitplane(0,3)

def bitplane_blue_4(event):
    bitplane(0,4)
    
def bitplane_blue_5(event):
    bitplane(0,5)
    
def bitplane_blue_6(event):
    bitplane(0,6)
    
def bitplane_blue_7(event):
    bitplane(0,7)
    
def bitplane_green_0(event):
    bitplane(1,0)
    
def bitplane_green_1(event):
    bitplane(1,1)
    
def bitplane_green_2(event):
    bitplane(1,2)
    
def bitplane_green_3(event):
    bitplane(1,3)
    
def bitplane_green_4(event):
    bitplane(1,4)
    
def bitplane_green_5(event):
    bitplane(1,5)
    
def bitplane_green_6(event):
    bitplane(1,6)
    
def bitplane_green_7(event):
    bitplane(1,7)
    
def bitplane_red_0(event):
    bitplane(2,0)
    
def bitplane_red_1(event):
    bitplane(2,1)
    
def bitplane_red_2(event):
    bitplane(2,2)
    
def bitplane_red_3(event):
    bitplane(2,3)
    
def bitplane_red_4(event):
    bitplane(2,4)
    
def bitplane_red_5(event):
    bitplane(2,5)
    
def bitplane_red_6(event):
    bitplane(2,6)
    
def bitplane_red_7(event):
    bitplane(2,7)
    
  
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
    btn_neg.grid(row = 0, column = 0)
    btn_neg.bind('<ButtonRelease-1>', neg_img)

    btn_bitplane_blue_7 = Button(
        master=frame,
        text="Blue 7"
    )
    btn_bitplane_blue_7.grid(row=1, column=0)
    btn_bitplane_blue_7.bind('<ButtonRelease-1>', bitplane_blue_7)

    btn_bitplane_blue_6 = Button(
        master=frame,
        text="Blue 6"
    )
    btn_bitplane_blue_6.grid(row=1, column=1)
    btn_bitplane_blue_6.bind('<ButtonRelease-1>', bitplane_blue_6)

    btn_bitplane_blue_5 = Button(
        master=frame,
        text="Blue 5"
    )
    btn_bitplane_blue_5.grid(row=1, column=2)
    btn_bitplane_blue_5.bind('<ButtonRelease-1>', bitplane_blue_5)

    btn_bitplane_blue_4 = Button(
        master=frame,
        text="Blue 4"
    )
    btn_bitplane_blue_4.grid(row=1, column=3)
    btn_bitplane_blue_4.bind('<ButtonRelease-1>', bitplane_blue_4)

    btn_bitplane_blue_3 = Button(
        master=frame,
        text="Blue 3"
    )
    btn_bitplane_blue_3.grid(row=1, column=4)
    btn_bitplane_blue_3.bind('<ButtonRelease-1>', bitplane_blue_3)

    btn_bitplane_blue_2 = Button(
        master=frame,
        text="Blue 2"
    )
    btn_bitplane_blue_2.grid(row=1, column=5)
    btn_bitplane_blue_2.bind('<ButtonRelease-1>', bitplane_blue_2)

    btn_bitplane_blue_1 = Button(
        master=frame,
        text="Blue 1"
    )
    btn_bitplane_blue_1.grid(row=1, column=6)
    btn_bitplane_blue_1.bind('<ButtonRelease-1>', bitplane_blue_1)

    btn_bitplane_blue_0 = Button(
        master=frame,
        text="Blue 0"
    )
    btn_bitplane_blue_0.grid(row=1, column=7)
    btn_bitplane_blue_0.bind('<ButtonRelease-1>', bitplane_blue_0)

    btn_bitplane_green_7 = Button(
        master=frame,
        text="Green 7"
    )
    btn_bitplane_green_7.grid(row=2, column=0)
    btn_bitplane_green_7.bind('<ButtonRelease-1>', bitplane_green_7)

    btn_bitplane_green_6 = Button(
        master=frame,
        text="Green 6"
    )
    btn_bitplane_green_6.grid(row=2, column=1)
    btn_bitplane_green_6.bind('<ButtonRelease-1>', bitplane_green_6)

    btn_bitplane_green_5 = Button(
        master=frame,
        text="Green 5"
    )
    btn_bitplane_green_5.grid(row=2, column=2)
    btn_bitplane_green_5.bind('<ButtonRelease-1>', bitplane_green_5)

    btn_bitplane_green_4 = Button(
        master=frame,
        text="Green 4"
    )
    btn_bitplane_green_4.grid(row=2, column=3)
    btn_bitplane_green_4.bind('<ButtonRelease-1>', bitplane_green_4)

    btn_bitplane_green_3 = Button(
        master=frame,
        text="Green 3"
    )
    btn_bitplane_green_3.grid(row=2, column=4)
    btn_bitplane_green_3.bind('<ButtonRelease-1>', bitplane_green_3)

    btn_bitplane_green_2 = Button(
        master=frame,
        text="Green 2"
    )
    btn_bitplane_green_2.grid(row=2, column=5)
    btn_bitplane_green_2.bind('<ButtonRelease-1>', bitplane_green_2)

    btn_bitplane_green_1 = Button(
        master=frame,
        text="Green 1"
    )
    btn_bitplane_green_1.grid(row=2, column=6)
    btn_bitplane_green_1.bind('<ButtonRelease-1>', bitplane_green_1)

    btn_bitplane_green_0 = Button(
        master=frame,
        text="Green 0"
    )
    btn_bitplane_green_0.grid(row=2, column=7)
    btn_bitplane_green_0.bind('<ButtonRelease-1>', bitplane_green_0)

    btn_bitplane_red_7 = Button(
        master=frame,
        text="Red 7"
    )
    btn_bitplane_red_7.grid(row=3, column=0)
    btn_bitplane_red_7.bind('<ButtonRelease-1>', bitplane_red_7)

    btn_bitplane_red_6 = Button(
        master=frame,
        text="Red 6"
    )
    btn_bitplane_red_6.grid(row=3, column=1)
    btn_bitplane_red_6.bind('<ButtonRelease-1>', bitplane_red_6)

    btn_bitplane_red_5 = Button(
        master=frame,
        text="Red 5"
    )
    btn_bitplane_red_5.grid(row=3, column=2)
    btn_bitplane_red_5.bind('<ButtonRelease-1>', bitplane_red_5)

    btn_bitplane_red_4 = Button(
        master=frame,
        text="Red 4"
    )
    btn_bitplane_red_4.grid(row=3, column=3)
    btn_bitplane_red_4.bind('<ButtonRelease-1>', bitplane_red_4)

    btn_bitplane_red_3 = Button(
        master=frame,
        text="Red 3"
    )
    btn_bitplane_red_3.grid(row=3, column=4)
    btn_bitplane_red_3.bind('<ButtonRelease-1>', bitplane_red_3)

    btn_bitplane_red_2 = Button(
        master=frame,
        text="Red 2"
    )
    btn_bitplane_red_2.grid(row=3, column=5)
    btn_bitplane_red_2.bind('<ButtonRelease-1>', bitplane_red_2)

    btn_bitplane_red_1 = Button(
        master=frame,
        text="Red 1"
    )
    btn_bitplane_red_1.grid(row=3, column=6)
    btn_bitplane_red_1.bind('<ButtonRelease-1>', bitplane_red_1)

    btn_bitplane_red_0 = Button(
        master=frame,
        text="Red 0"
    )
    btn_bitplane_red_0.grid(row=3, column=7)
    btn_bitplane_red_0.bind('<ButtonRelease-1>', bitplane_red_0)
    
    
    # button for select image
    btn_select = Button(
        master = frame,
        text = "Select an Image",
        underline = 0
    )
    btn_select.grid(row = 4, column = 0)
    btn_select.bind('<ButtonRelease-1>', select_image)
    

    # Bind all the required keys to functions

    root.bind("<q>", quit_img)
    
    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()
