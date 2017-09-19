import cv2
import numpy as np
from utils import *
from ImageTransform import *

class VideoStabilization:
    def __init__(self, src_video_file_path, dst_video_file_path):
        cap = cv2.VideoCapture(src_video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 0  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 0# float
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(dst_video_file_path + '.avi',fourcc, fps, (width,height))

        transform_image = ImageTransform(False)
        it = 0
        first_frame = None
        print("Skipping first 30 frames...")
        while(cap.isOpened()):
            #print('in')
            ret, frame = cap.read()
            if ret==True:
                
                it += 1
                if it < 30:
                    continue

                print("frame", it-30, "of", length-30, ', %.2f %%' % (100*((it-30)/(length-30))))
                
                out_frame = None
                if first_frame is not None:
                    #first_frame is already with a border     
                    newframe = first_frame.copy()               
                    out_frame = transform_image.show_matched_reuse(newframe, frame)
                    if out_frame is None:
                        print("skipping frame")
                        continue                                                 
                else:
                    out_frame = frame.copy()
                    first_frame = frame.copy()
                    new_frame = frame
                    transform_image.set_first_frame(new_frame)

                debug('first frame', first_frame)
                debug('out frame', out_frame)

                w, h, c = out_frame.shape
                out.write(out_frame)
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()