from tkinter import Tk, filedialog, Frame, Label, Button
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import filetype
import math
from fractions import Fraction


# set up an a list for files
images = []
# to store the image file types
filetypes = []


# Load the image referred to by path
def opencv_img(path):
    # read and convert image
    image = cv2.imread(path)
    image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
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
    
# Add c to each pixel in the image  
def add_c(c):
    global image
    new_image = image.copy()
    new_image[new_image + c >= new_image] += c
    new_image[new_image + c < new_image] = 255
    update_new(new_image)
    
# Subtract c from each pixel in the image    
def minus_c(c):
    global image
    new_image = image.copy()
    new_image[new_image - c <= new_image] -= c
    new_image[new_image - c > new_image] = 0
    update_new(new_image)
    
# Multilply each pixel in the image by c
def times_c(c):
    global image
    new_image = image.copy()
    # For integer operations
    frac = Fraction(c).limit_denominator()
    mult = frac.numerator
    divide = frac.denominator
    
    #Scaling up
    if mult >=1  and divide >= 1:
        new_image[new_image * mult >= new_image] *= mult
        new_image //= divide
        new_image[new_image * mult  < new_image] = 255
        
    else:  #scale down
        new_image //= divide
        
    update_new(new_image)
 
# Divide each pixel by c       
def divide_c(c):
    global image
    new_image = image.copy()
    new_image //= c
    update_new(new_image)
    

def prompt_arithmetic(event):
    operations = ["add", "subtract", "multiply", "divide"]
    short = ["+", "-", "*", "/"]
    
    options = {"+" : add_c, "add" : add_c,
               "-" : minus_c, "subtract" : minus_c,
               "*" : times_c, "multiply" : times_c,
               "/" : divide_c, "divide": divide_c,}
    
    
    while(True):
        op = simpledialog.askstring("Input", "What operation? (+,-,*,/)",
                                       parent=root)
        if op != None and (op.lower() in operations or op in short):
            break
    if op.lower() in [ "*", "multiply"]:
        while (True):
            c = simpledialog.askfloat("Input", "What C?",
                                             parent=root)
            if c != None:
                break
   
    else:
        while (True):
            c = simpledialog.askinteger("Input", "What C?",
                                             parent=root)
            if c != None:
                break

        

    options[op.lower()](c)
    


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

# Prompt user for Piecewise Linear transformation points
def prompt_plinear(event):
    
    while(True):
        r1 = simpledialog.askinteger("Input", "For the point (r1, s1), enter an integer r1  from [1,254]", 
                                    parent=root, 
                                    minvalue=1, maxvalue=254)
        if r1 != None:
            break
        
    while(True):
        s1 = simpledialog.askinteger("Input", "For the point (r1, s1), enter an integer s1 from [0,255]",
                                    parent=root,
                                    minvalue=0, maxvalue=255)
        if s1 != None:
            break
        
    if (int(r1) < 254):
        while(True):
            r2 = simpledialog.askinteger("Input", "For the point (r2, s2), enter a integer r2  from (" + 
                                        str(r1+1) + " , 255]", 
                                        parent=root, 
                                        minvalue=(r1 + 1), maxvalue=255)
            if r2 != None:
                break
    else:
        r2 = 255 
        
    while(True):
        s2 = simpledialog.askinteger("Input", "For the point (r2, s2), enter an integer s2 from (" + 
                                    str(s1) + " , 255]", 
                                    parent=root, minvalue=0, maxvalue=255)
        if s2 != None:
            break   
        
        
    piecewise_linear(r1, s1, r2, s2) 
    
# Piecewise Linear Transformation of image   
def piecewise_linear(r1, s1, r2, s2):
    global image
    
    # Faster numpy trick
    plinear_img = image.copy()
    
    
    # need two input points (r1, s1) and (r2, s2)
    # where  0 < r1 < r2 <  255 and 
    # and 0 <= s1 <= s2 <= 255 
    # for pixel values in range [0, r1) we apply the linear
    # transformation having slope (s1- 0)/(r1-0) and intercept 0
    # for pixel values in range [r1, r2) we use the linear
    # transformation with slope (s2 - s1)/(r2-r1) and intercept 0
    # pixel values greater than or equal to r2 are acted on by
    # the linear transformation (255 - s2)/(255 - r2)
    
    plinear_img[plinear_img < r1] *= s1//r1
    plinear_img[plinear_img >= r2] *= (255 - s2)//(255 - r2)
    plinear_img[(plinear_img >= r1) <r2 ] *=  (s2 - s1)//(r2 - r1)
    
    plinear_img = np.array(plinear_img, dtype = np.uint8)
    update_new(plinear_img)

# Prompt User for threshold value so that any value below that value
# is taken to 0 and any value at least the threshold is taken to max
def prompt_threshold(event):
    while(True):
        thresh = simpledialog.askinteger("Input", "Enter an integer threshold value from [0,255]", 
                                    parent=root, 
                                    minvalue=0, maxvalue=255)
        if thresh != None:
            break
    
    while(True):
        newmax = simpledialog.askinteger("Input", "Enter a max integer pixel value in [ " + str(thresh) + ", 255]",
                                    parent=root, 
                                    minvalue= thresh, maxvalue=255)
        if thresh != None:
            break
                                         
    threshold(thresh, newmax)
    
def threshold(tvalue, maxvalue):
    global image
    
    thresh_img = image.copy()
    
    thresh_img[thresh_img < tvalue] = 0
    thresh_img[thresh_img >= tvalue] = maxvalue
    
    thresh_img = np.array(thresh_img, dtype = np.uint8)
    update_new(thresh_img)    

# Show the image chosen by the user
def update_original(path):
    global original, image
    image = opencv_img(path)
    disp_img = convert_img(image)

    original.configure(image=disp_img)
    original.image = disp_img
    return disp_img

# Show the transformed image 
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
    
     
    btn_arithmetic = Button(
        master = frame,
        text = "Arithmetic",
        underline = 0
    )
    btn_arithmetic.grid(row = 0, column = 3)
    btn_arithmetic.bind('<ButtonRelease-1>', prompt_arithmetic)
    
    # Button for binarization/threshold
    btn_threshold = Button(
        master = frame,
        text = "Binarization/Threshold",
        underline = 0
    )
    btn_threshold.grid(row = 0, column = 4)
    btn_threshold.bind('<ButtonRelease-1>', prompt_threshold)

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

    # button for bitplane
    btn_plinear = Button(
        master = frame,
        text = "Piecewise Linear",
        underline = 0
    )
    btn_plinear.grid(row = 0, column = 3)
    btn_plinear.bind('<ButtonRelease-1>', prompt_plinear)


    # Bind all the required keys to functions

    root.bind("<q>", quit_img)

    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()
