# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tkinter import Tk, filedialog, Frame, Label, Button, simpledialog, filedialog
from PIL import Image, ImageTk
from time import time
from tkinter.messagebox import showinfo, askyesno
import argparse
import cv2
import filetype
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from fractions import Fraction

# Flag for determining if images are loaded
img = False
second_img = False

#Flags for determining if the image is grayscal (default is true)
img_gray = True
img2_gray = True

#To reload image 2 in the event of a type mismatch
img2_path = ""
##-------Launching the program-----------------------------------------------##
# For -h argument
def get_args():
    parser = argparse.ArgumentParser(description='Pixel Operations v1.0')
    parser.add_argument('-- ', default=" ",
                        help='A Graphical User Interface with no arguments.')
    args = parser.parse_args()
    return(args)

##-------Functions to open/read an image file and rendering in UI------------##

# Read in image referred to by path and conform to fit screen
def opencv_img(path):
    # read and convert image
    image = cv2.imread(path)
    defaultrows = 420
    defaultcolumn = 580
    # Set scale multiplier to the lowest of the following values:
    # 1
    # window row count / image row count
    # window column count / image column count
    scale = min(1, min(defaultrows / image.shape[0], defaultcolumn / image.shape[1]))

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


##----------User Controls for the UI-----------------------------------------##

# Select the image to load
def select_img1(event):
    global img, second_img
    # Prompt the user
    path = filedialog.askopenfilename(title="Please Select Image 1")
    # if there is a path and it is readable
    if len(path) > 0 and cv2.haveImageReader(path):
        color_img()
        update_img1(path)
        img = True
        if second_img:
            correct_mismatch()
            get_subsets()
    else:
        showinfo("Error", "No Image was found at path or it is not readable.")

def select_img2(event):
    global second_img
    
    if img == False:
        showinfo("Error", "Load Image 1 First")
        return
    # Prompt the user
    path = filedialog.askopenfilename(title="Please Select Image 2")
    # if there is a path and it is readable
    if len(path) > 0 and cv2.haveImageReader(path):
        update_img2(path)
        get_subsets()
        second_img = True
    else:
        showinfo("Error", "No image was found at path or it is not readable.")

#Exit the program
def quit_img(event):
    root.destroy() #Kill the display
    sys.exit(0)
    
# Save the image to the main given path appending the name of any transformation
def save_img(event):
    name = filedialog.asksaveasfilename(confirmoverwrite=True)
    # If no name or an invalid name is given, cancel
    if name == None or name[0] == ".":
        return
    #Default file type if none is given
    if "." not in name:
        name = name+".png"
    cv2.imwrite(name, new_img)

#Determine if the image should be color or grayscale
def color_img():
    global img_gray, img2_gray
    answer = askyesno("Question","Default is Grayscale image. Do you want to use a Color Image?")
    if answer:
        img_gray = False
        img2_gray = False
    else: 
        img_gray = True
        img2_gray = True

#If image 1 is changed to grayscale or color, convert image 2 to match image 1  
def correct_mismatch():
    global img2_path
    showinfo("Updating", "Updating Image 2 to match Image 1 format")
    update_img2(img2_path)
        

##---------GUI update image formating ---------------------------------------##
# User given path to image, open and format image return disp_img        
def update_img1(path):
    global img1, image
    #Load the image
    image = opencv_img(path)
    #Convert to grayscale if requested
    if img_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Convert and display
    disp_img = convert_img(image) 
    img1.configure(image=disp_img)
    img1.image = disp_img
    return disp_img

def update_img2(path):
    global img2, image2, img2_path
    #Load the image
    img2_path = path
    image2 = opencv_img(path)
    #Convert to grayscale if requested
    if img2_gray: 
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #Convert and display
    disp_img = convert_img(image2)
    img2.configure(image=disp_img)
    img2.image = disp_img
    return disp_img

# Create subsets of images 1 & 2 of size the smaller of the two widths and two heights
def get_subsets():
    global image, image2, img1_subset, img2_subset
    img1_subset = image[0:int(min(image.shape[0], image2.shape[0])),
                        0:int(min(image.shape[1], image2.shape[1]))]
   
    img2_subset = image2[0:int(min(image.shape[0], image2.shape[0])),
                         0:int(min(image.shape[1], image2.shape[1]))]
   

# A newly transformed image, new, is formatted for display
def update_new(img):      
    global new, new_img
    new_img = img
    disp_img = convert_img(img)
    new.configure(image=disp_img)
    new.image = disp_img
    
  
# Check if the first image is loaded
def is_image():
    global img   
    #Check that image 1 is loaded
    if not img:
        showinfo("Error", "Image 1 has not been selected. Please select image 1.")
        return False
    return True

##---------Pixel Transformations---------------------------------------------##

   
# Negative Transformation of image 1
def neg_img(event):
    global image    
    #Check that image 1 is loaded
    if not is_image():
        return
    
    neg_img = 255-image
    
    #Update the transformation window
    update_new(neg_img)

# Bitplane Prompt for user
def prompt_bitplane(event):
    
    #Check that image 1 is loaded
    if not is_image():
        return
    
    colors = ["blue", "green", "red"]
    
    #Is the image color?
    if img_gray == False:
        # Get the color to use
        while(True):
            color = simpledialog.askstring("Input", "What color? (red, green, or blue)",
                                           parent=root)
            if color == None:
                return 
            # Break if acceptable, else ask again
            elif color.lower() in colors:
                color_code = colors.index(color.lower())
                break
    else:
        color_code = None
    
    # Get the bit value
    bit = simpledialog.askinteger("Input", "What bit value? (0-7)",
                                     parent=root,
                                     minvalue=0, maxvalue=7)
    # If user cancels
    if bit == None:
        return
    #Perform the tranformation
    bitplane(bit, color_code)
 
# Bitplane Transformaton of image    
def bitplane(bit, color=None):
    global image
    
    if img_gray:
        #Bitplane for grayscale
        bitplane_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        bitplane_img[:, :][image[:, :] % 2 ** (bit + 1) >= 2 ** bit] = np.uint8(255)
    else:
        # For color
        bitplane_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        bitplane_img[:,:,color][image[:,:,color]% 2**(bit+1) >= 2**bit] = np.uint8(255)

     #Update the transformation window
    update_new(bitplane_img)

# Prompt user for Arithmetic Operations
def prompt_arithmetic(event):
    
    #Check that image 1 is loaded
    if not is_image():
        return
    
    # List of allowed operations
    operations = ["add", "subtract", "multiply", "divide", "+", "-", "*", "/"]
    
    options = {"+" : add_c, "add" : add_c,
               "-" : minus_c, "subtract" : minus_c,
               "*" : times_c, "multiply" : times_c,
               "/" : divide_c, "divide": divide_c,}
    
    # Prompt the user for an operation
    while(True):
        op = simpledialog.askstring("Input", "Which Arithmetic operation? Enter one: +, -, *, /",
                                       parent=root)
        # User enters nothing or cancels
        if  op == None:
            return
        # Break if acceptable, else ask again
        elif op.lower() in operations:
            break
    
    if op.lower() in [ "*", "multiply"]:
        c = simpledialog.askfloat("Input", "Please enter a non-negative value for the mulitplier c.",
                                  parent=root, minvalue=0.0)
        # User enters nothing or cancels
        if c == None:
            return
    
    elif op.lower() in [ "/", "divide"]:
        c = simpledialog.askfloat("Input", "Please enter a positive value for the divisor c",
                                  parent=root, minvalue=0.1)
        # User enters nothing or cancels
        if c == None:
            return
   
    else:
        c = simpledialog.askinteger("Input", "Please enter a non-negative integer value for the constant c.",
                                         parent=root, minvalue=0)
        # User enters nothing or cancels
        if c == None:
            return
                 
    # Perform the user chosen operation
    options[op.lower()](c)
        
    
# Arithmetic Add c to each pixel in the image  
def add_c(c):
    global image
    
    new_image = image.copy()
    new_image[new_image + c >= new_image] += c
    new_image[new_image + c < new_image] = 255
    #Update the transformation window
    update_new(new_image)
    
# Arithmetic Subtract c from each pixel in the image    
def minus_c(c):
    global image
    
    new_image = image.copy()
    new_image[new_image - c <= new_image] -= c
    new_image[new_image - c > new_image] = 0
    #Update the transformation window
    update_new(new_image)
    
# Arithmetic Multilply each pixel in the image by c
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
    #Update the transformation window
    update_new(new_image)
 
# Arithmetic Divide each pixel by c       
def divide_c(c):
    global image
    new_image = image.copy()
    new_image //= c
    #Update the transformation window
    update_new(new_image)
    

# Logarithmic Transformation of image    
def log_trans(event):
    global image
    
    #Check that image 1 is loaded
    if not is_image():
        return
    
    cpy_img = image.copy()
    
    # Add 1 to all pixel values except those at 255 to prevent overflow
    cpy_img[cpy_img<255] += 1
    log = np.log(cpy_img)
   
    #Log transformation
    log_img = (255/np.max(log)) * log
    log_img = np.array(log_img, dtype = np.uint8)
    
    #Update the transformation window
    update_new(log_img)
    

# Prompt user for Piecewise Linear transformation points
def prompt_plinear(event):
    
    #Check that image 1 is loaded
    if not is_image():
        return
    
    r1 = simpledialog.askinteger("Input", "For the point (r1, s1), enter r1  from (0,254]", 
                                parent=root, 
                                minvalue=1, maxvalue=254)
    # User enters nothing or cancels
    if r1 == None:
        return
        
    s1 = simpledialog.askinteger("Input", "For the point (r1, s1), enter s1 from [0,255]",
                                parent=root, minvalue=0, maxvalue=255)
    # User enters nothing or cancels
    if s1 == None:
        return
        
    if (int(r1) < 254):
        r2 = simpledialog.askinteger("Input", "For the point (r2, s2), enter r2  from (" + 
                                    str(r1+1) + " , 255]", 
                                    parent=root, 
                                    minvalue=(r1 + 1), maxvalue=255)
        # User enters nothing or cancels
        if r2 == None:
            return
    else:
        r2 = 255 

    s2 = simpledialog.askinteger("Input", "For the point (r2, s2), enter s2 from (" + 
                                str(s1) + " , 255]", 
                                parent=root, minvalue=0, maxvalue=255)
    # User enters nothing or cancels
    if s2 == None:
        return   
        
    # Perform the transformation  
    piecewise_linear(r1, s1, r2, s2) 
    
# Piecewise Linear Transformation of image   
def piecewise_linear(r1, s1, r2, s2):
    global image
    
    #Check that image 1 is loaded
    if not is_image():
        return
   
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
    
    plinear_img[plinear_img < r1] *= np.uint8(s1//r1)
    plinear_img[plinear_img >= r2] *= np.uint8((255 - s2)//(255 - r2))
    plinear_img[(plinear_img >= r1) <r2 ] *=  np.uint8((s2 - s1)//(r2 - r1))
    
    plinear_img = np.array(plinear_img, dtype = np.uint8)
    
    #Update the transformation window
    update_new(plinear_img)
 
# Prompt User for threshold value so that any value below that value
# is taken to 0 and any value at least the threshold is taken to max
def prompt_threshold(event):
    global image
    
    #Check that image 1 is loaded
    if not is_image():
        return

    thresh = simpledialog.askinteger("Input", "Enter an integer threshold value from [0,255]", 
                                parent=root, 
                                minvalue=0, maxvalue=255)
    
    # User enters nothing or cancels
    if thresh == None:
        return
    
    newmax = simpledialog.askinteger("Input", "Enter a max integer pixel value in [ " + str(thresh) + ", 255]",
                                parent=root, 
                                minvalue= thresh, maxvalue=255)
    # User enters nothing or cancels
    if newmax == None:
        return
    
    # Perform the transformation                                  
    threshold(thresh, newmax, image)
    
    
# Input image allows threshold to be called by other functions other than user button 
# and applied to only global image   
def threshold(tvalue, maxvalue, image):

    thresh_img = image.copy()
    
    thresh_img[thresh_img < tvalue] = 0
    thresh_img[thresh_img >= tvalue] = maxvalue
    
    thresh_img = np.array(thresh_img, dtype = np.uint8)
     #Update the transformation window
    update_new(thresh_img)
    return thresh_img

def prompt_gamma(event):
    
    #Check an image is loaded
    if not is_image():
        return
    
   
    gvalue = simpledialog.askfloat("Input", "Output pixel will be c(pixel)^gamma. Please enter in a float value for the exponent gamma at least 0",
                                     parent=root,
                                     minvalue = 0.0)
    if gvalue == None:
        return

   
    cvalue = simpledialog.askfloat("Input", "Output pixel will be c(pixel)^gamma. Please enter in a float multiplier c at least 0",
                                     parent=root,
                                     minvalue = 0.0)
    if cvalue == None:
        return     
     
    # Perform the gamma transformation                                  
    gamma_trans(gvalue, cvalue)
    
      
def gamma_trans(gamma, multiplier):
    gamma_img = np.array(image, dtype=np.float32)
    
    # breaking down into integer operations to raise pixel datatypes to power of gamma
    gamma_img /= 255
    gamma_img **= gamma
    gamma_img *= (multiplier * 255)
    
    # translate back to uint8 datatype
    gamma_img = np.array(gamma_img, dtype = np.uint8)
    
    # update the transformation window
    update_new(gamma_img)

#Prompt the user for what binary set operation they want.    
def prompt_set(event):
    
    #Check an image is loaded
    if not is_image():
        return
    
    #Requires two images
    if not second_img:
        showinfo("Error", "Binary Set Operations require two images.  Load image 2 first and try again")
        return
    
    #Allowed operations
    operations = ["union", "intersection", "difference", "u", "i", "d"]
    options = {"union" : union, "u" : union,
               "intersection" : intersection, "i" : intersection,
               "difference" : difference, "d" : difference}
    
    # Propmt the user for an operation
    while(True):
        op = simpledialog.askstring("Input", "Union, Intersection, or Difference?",
                                       parent=root)
        
        # User enters nothing or cancels
        if  op == None:
            return
        
        # Break if acceptable, else ask again
        elif op.lower() in operations:
            break
        
    # Call the user chosen operation
    options[op.lower()]()
    
# Union of the current two images   
def union():
    global img1_subset, img2_subset
    new = np.maximum(img1_subset, img2_subset)
     #Update the transformation window
    update_new(new)

# Intersection of the current two images  
def intersection():
    global image, image2
    new = np.minimum(img1_subset, img2_subset)
     #Update the transformation window
    update_new(new)

# Set difference of the current two images
def difference():
    global img1_subset, img2_subset
    new = img1_subset.copy()
    new[new == img2_subset] = 0 
     #Update the transformation window
    update_new(new)
    
# The complement of the current image using C
def complement(event):
    global image    
    
    #Check an image is loaded
    if not is_image():
        return
    
    new = image.copy()
    
    # Prompt the user for the value of C
    c= simpledialog.askinteger("Input", "New pixel = min of pixel - c and 0. Enter a positive integer value for the constant c.", 
                                parent=root, 
                                minvalue=0)
    # User enters nothing or cancels
    if c == None:
        return
    
    new[new - c <= new] -= c
    new[new - c > new] = 0
     #Update the transformation window
    update_new(new)

#Prompt the user for what binary set operation they want.    
def prompt_logic(event):
    
    #Check an image is loaded
    if not is_image():
        return
    
    #Requires two images
    #if not second_img:
    #    showinfo("Error", "Bitwise Logic Operations require two images.  Load image 2 first and try again")
    #    return
    
    
    #Allowed logical operations
    operations = ["and", "or", "xor", "not", "a", "o", "x", "n"]
    options = {"and": bitwise_and, "a" : bitwise_and,
               "or" : bitwise_or, "o" : bitwise_or,
               "xor" : bitwise_xor, "x" : bitwise_xor,
               "not": bitwise_not, "n": bitwise_not}

    # Propmt the user for an operation
    while(True):
        op = simpledialog.askstring("Input",
                                    "For logical bitwise operations, please enter: and, or, xor",
                                     parent=root)
        # Make sure they answer
        if op != None and (op.lower() in operations):
            break
        if not second_img and (op.lower() in operations):
            showinfo("Error", "Logical operations require two images. Please select a second image and then try again.")
            return
    
    # Call the user chosen operation
    options[op.lower()]()  
    
def bitwise_and():
    #Requires two images
    if not second_img:
        showinfo("Error", "Bitwise Logic Operations require two images.  Load image 2 first and try again")
        return
    
    thresh_img1 = threshold(127, 255, img1_subset)
    thresh_img2 = threshold(127, 255, img2_subset)
    and_img = cv2.bitwise_and(thresh_img1, thresh_img2)
     #Update the transformation window
    update_new(and_img)

def bitwise_or():
    
    #Requires two images
    if not second_img:
        showinfo("Error", "Bitwise Logic Operations require two images.  Load image 2 first and try again")
        return
    
    thresh_img1 = threshold(127, 255, img1_subset)
    thresh_img2 = threshold(127, 255, img2_subset)
    or_img = cv2.bitwise_or(thresh_img1, thresh_img2)
     #Update the transformation window
    update_new(or_img)

def bitwise_xor():
    #Requires two images
    if not second_img:
        showinfo("Error", "Bitwise Logic Operations require two images.  Load image 2 first and try again")
        return
    
    
    thresh_img1 = threshold(127, 255, img1_subset)
    thresh_img2 = threshold(127, 255, img2_subset)
    xor_img = cv2.bitwise_xor(thresh_img1, thresh_img2)
    #Update the transformation window
    update_new(xor_img)

def bitwise_not():
    thresh_img = threshold(127, 255, image)
    not_img = cv2.bitwise_not(thresh_img)
    #Update the transformation window
    update_new(not_img)    


##---------------------------------------------------------------------------##
def main():
    global root, img1, img2, img1_subset, img2_subset, new, image, image2

    #Get the command arguments
    get_args()
    if len(sys.argv) != 1:
        print('Pixel Transformation Operations v1.0 is a GUI program without arguments')
        sys.exit(0)
        
    root = Tk()
    root.title("Pixel Transformations. To begin load image 1.")
    
    # The original loaded image
    img1 = Label(image=None)
    img1.pack(side="left", padx=10, pady=10)

    img2 = Label(image=None)
    img2.pack(side="left", padx=10, pady=10)
    
    # The new modifed image
    new = Label(image=None)
    new.pack(side="right", padx=10, pady=10)

    # Frame to display navigation buttons at bottom of window
    frame = Frame()
    frame.pack()
 
     
      
    # Button for select image
    btn_select_img1 = Button(
        master = frame,
        text = "Select image 1",
        underline = 13
    )
    btn_select_img1.grid(row = 0, column = 1)
    btn_select_img1.bind('<ButtonRelease-1>', select_img1)

   
     # Button for log transformation of image
    btn_log = Button(
        master = frame,
        text = "Log",
        underline = 0
    )
    btn_log.grid(row = 2, column = 0)
    btn_log.bind('<ButtonRelease-1>', log_trans)
    
    # button for piecewise linear
    btn_gamma = Button(
        master = frame,
        text = "Gamma",
        underline = 0
    )
    btn_gamma.grid(row = 2, column = 2)
    btn_gamma.bind('<ButtonRelease-1>', prompt_gamma)
    
    # Button for negative of image
    btn_neg = Button(
        master = frame,
        text = "Negative",
        underline = 0
    )
    btn_neg.grid(row = 6, column = 2)
    btn_neg.bind('<ButtonRelease-1>', neg_img)
    
    
    # Button for Arithmetic
    btn_arithmetic = Button(
        master = frame,
        text = "Arithmetic",
        underline = 0
    )
    btn_arithmetic.grid(row = 6, column = 0)
    btn_arithmetic.bind('<ButtonRelease-1>', prompt_arithmetic)
    
    # Button for binarization/threshold
    btn_threshold = Button(
        master = frame,
        text = "Binarization/Threshold",
        underline = 13
    )
    btn_threshold.grid(row = 8, column = 0)
    btn_threshold.bind('<ButtonRelease-1>', prompt_threshold)
    
    # Button for bitplane
    btn_bit = Button(
        master = frame,
        text = "Bitplane",
        underline = 0
    )
    btn_bit.grid(row = 4, column = 0)
    btn_bit.bind('<ButtonRelease-1>', prompt_bitplane)
    
    
   
    btn_comp = Button(
        master=frame,
        text="Complement",
        underline = 0
    )
    btn_comp.grid(row=4, column=2)
    btn_comp.bind('<ButtonRelease-1>', complement)
    
    
    # button for piecewise linear
    btn_plinear = Button(
        master = frame,
        text = "Piecewise Linear",
        underline = 0
    )
    btn_plinear.grid(row = 8, column = 2)
    btn_plinear.bind('<ButtonRelease-1>', prompt_plinear)
    
    


    btn_select_img2 = Button(
        master=frame,
        text="Select image 2",
        underline=13
    )
    btn_select_img2.grid(row=14, column=1)
    btn_select_img2.bind('<ButtonRelease-1>', select_img2)
    
    btn_bin = Button(
        master=frame,
        text="Binary Set Operations"
    )
    btn_bin.grid(row=16, column=0)
    btn_bin.bind('<ButtonRelease-1>', prompt_set)
    
    btn_bin = Button(
        master=frame,
        text="Bitwise Logic Operations"
    )
    btn_bin.grid(row=16, column=2)
    btn_bin.bind('<ButtonRelease-1>', prompt_logic)
    
    # btn_bitwise_and = Button(
    #     master=frame,
    #     text="AND",
    # )
    # btn_bitwise_and.grid(row=14, column=0)
    # btn_bitwise_and.bind('<ButtonRelease-1>', bitwise_and)

    # btn_bitwise_or = Button(
    #     master=frame,
    #     text="OR",
    # )
    # btn_bitwise_or.grid(row=14, column=1)
    # btn_bitwise_or.bind('<ButtonRelease-1>', bitwise_or)

    # btn_bitwise_xor = Button(
    #     master=frame,
    #     text="XOR",
    # )
    # btn_bitwise_xor.grid(row=14, column=2)
    # btn_bitwise_xor.bind('<ButtonRelease-1>', bitwise_xor)

    # btn_bitwise_not = Button(
    #     master=frame,
    #     text="NOT",
    # )
    # btn_bitwise_not.grid(row=14, column=3)
    # btn_bitwise_not.bind('<ButtonRelease-1>', bitwise_not)
 
   
    
    
   
    # Button for save_img image
    btn_save = Button(
        master = frame,
        text = "Save",
        underline = 0
    )
    btn_save.grid(row = 18, column = 1)
    btn_save.bind('<ButtonRelease-1>', save_img)
    
    # Bind all the required keys to functions
    root.bind("<q>", quit_img)
    root.bind("<s>", save_img)
    root.bind("<S>", save_img)
    root.bind("<A>", prompt_arithmetic)
    root.bind("<N>", neg_img)
    root.bind("<B>", prompt_bitplane)
    root.bind("<G>", prompt_gamma)
    root.bind("<P>", prompt_plinear)
    root.bind("<L>", log_trans)
    
    
   

    root.mainloop() # Start the GUI
    
if __name__ == "__main__":
    main()
    

