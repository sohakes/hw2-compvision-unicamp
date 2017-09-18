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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 250  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 250# float
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print('width, height', width, height)
        out = cv2.VideoWriter(dst_video_file_path + '.avi',fourcc, fps, (width,height))
        out2 = cv2.VideoWriter(dst_video_file_path + 'unfix.avi',fourcc, fps, (width-250,height-250))
        #print('fps', fps)

        transform_image = ImageTransform()
        it = 0
        first_frame = None
        while(cap.isOpened()):
            #print('in')
            ret, frame = cap.read()
            if ret==True:
                print("frame", it, "of", length, ', %.2f %%' % (100*(it/length)))
                it += 1
                out_frame = None
                if first_frame is not None:
                    #first_frame is already with a border                    
                    out_frame = transform_image.show_matched_reuse(first_frame, frame)                                                 
                else:
                    #transform_image.set_first_frame(first_frame)                     
                    out_frame = cv2.copyMakeBorder(frame, 125, 125, 125, 125, cv2.BORDER_CONSTANT, 0)
                    first_frame = frame
                    transform_image.set_first_frame(first_frame)

                w, h, c = out_frame.shape
                #fout_frame = out_frame[200:w-200,200:h-200,:]
                #print('shapes outframe, frame', out_frame.shape, frame.shape)
                # write the flipped frame
                out.write(out_frame)
                out2.write(frame)

                #cv2.imshow('out_frame',fout_frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        out2.release()
        cv2.destroyAllWindows()