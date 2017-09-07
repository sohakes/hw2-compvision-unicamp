import cv2
import numpy as np
import math

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

DEBUG = True
NUMBER_FILE = -1

def numFile():
    global NUMBER_FILE
    NUMBER_FILE = NUMBER_FILE + 1
    return NUMBER_FILE


def debug_print(val):
   if DEBUG == False:
        return
   print(val)

def debug(name,img):
    if DEBUG == False:
        return
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gaussian_mask(size, sigma):
    assert size % 2 == 1, "mask should have odd size"
    def pixel_val(x):
        exp = math.e**(-(x**2)/(2*(sigma**2)))
        return ((1.0 * exp) /(math.sqrt(2 * math.pi) * sigma))

    halfsize_i = math.floor(size / 2)

    mask = np.array([pixel_val(i) for i in range(-halfsize_i, halfsize_i + 1)])
    print(mask)
    msum = np.sum(mask)

    return mask / msum

