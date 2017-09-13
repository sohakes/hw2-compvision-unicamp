import cv2
import numpy as np
from utils import *
from OpenCVImageTransform import *

class OpenCVVideoStabilization:
    def __init__(self, src_video_file_path, dst_video_file_path):
        cap = cv2.VideoCapture(src_video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        print(width, height)
        out = cv2.VideoWriter(dst_video_file_path + '.avi',fourcc, fps, (width,height))
        print(fps)

        transform_image = OpenCVImageTransform()

        last_frame = None
        while(cap.isOpened()):
            print('in')
            ret, frame = cap.read()
            if ret==True:
                out_frame = cv2.copyMakeBorder(frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
                if last_frame is not None:
                    #last_frame is already with a border
                    out_frame = transform_image.show_matched(last_frame, out_frame)
                                    
                last_frame = out_frame
                w, h, c = out_frame.shape
                fout_frame = out_frame[200:w-200,200:h-200,:]
                print(out_frame.shape, fout_frame.shape, frame.shape)
                # write the flipped frame
                print('out')
                out.write(fout_frame)

                #cv2.imshow('out_frame',fout_frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()