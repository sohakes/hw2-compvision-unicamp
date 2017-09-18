import cv2
import numpy as np
import math

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

DEBUG = False
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

