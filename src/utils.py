import cv2
import numpy as np
import math

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

DEBUG = False
NUMBER_FILE = -1
QUESTION = [0, 0, 0, 0]


def write_image(question, img, save):
    if save:
        cv2.imwrite('output/p2-'+str(question)+'-'+str(QUESTION[question-1]) + '.png', img)
        QUESTION[question-1] += 1

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

