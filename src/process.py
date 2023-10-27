import cv2
import time
import datetime

import logproc


class Methods(object):
    """
    Define model/flow for DeepStream used.

    An example for saved in the config/cfg4ds.json:
    {
        ...

        "your_customized_paramters":
        {
            
        }
    }
    """
    def __init__(self, local_args: dict, gpu_index: int):
        self.__local_args = local_args.copy()
    
        self.__flow_logger = logproc.Logger(self.__local_args["kernel"]["log"]["saved_dir"],
                                          self.__local_args["kernel"]["log"]["title_name"])
        
        # define/process/load your parameters/models in this area.

    def dispose(self, input_info: dict) -> dict:
        """
        flow processing, such as DL/ML model inference.

        basic input for input_info dict:
        { 
            "rgba_img": numpy.ndarray,
            "stream_fps": float,
            "debug_mode": bool,
        }

        return: 
        { 
            "detection_eqrectimg": numpy.ndarray, 
            "postproc_eqrectimg": numpy.ndarray
        }
        """
        rgba_img = input_info["rgba_img"]
        stream_fps = input_info["stream_fps"]
        debug_mode = input_info["debug_mode"]
        
        print("This source with prediction shape " + str(rgba_img.shape) + " & " + str(stream_fps) + " FPS.")

        # your processing.
        detectimg = None if not debug_mode else cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
        postimg = None if not debug_mode else cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)

        # your return parameters saved in the dict.
        draw_detect_info = \
        { 
            "detection_eqrectimg": detectimg, 
            "postproc_eqrectimg": postimg
        }


        return draw_detect_info