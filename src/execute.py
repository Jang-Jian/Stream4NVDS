import os
import sys
import json
import argparse

import logproc
import streamer


def loadcfg4ds(json_cfg_file: str) -> dict:
    """
    parse the contents of config for deepstream.
    """
    json_args = None
    with open(json_cfg_file, newline='') as jsonfile:
        json_args = json.load(jsonfile)
    return json_args


if __name__ == '__main__': 
    argparser = argparse.ArgumentParser(description = 'CAMRTSPDS - A Temaple for Scenario2.') 

    argparser.add_argument('-g', '--gpu_index', type=int,
                           help="Allocated gpu index for execution.", 
                           default=0)
    argparser.add_argument('-c', '--config_path', type=str,
                           help="A directory for config with json.", 
                           default="../config/deepstream.json")
    argparser.add_argument('-s','--log_saved_dir', type=str, 
                           help="A saved directory of log file.", 
                           default="./log")
    argparser.add_argument('-t','--log_title_name', type=str, 
                           help="A named file of log file.", 
                           default="myds")
    argparser.add_argument('-r','--rtsp_urls', type=str, 
                           help="Deliever urls for rtsp.", required=True)

    args = argparser.parse_args()
    main_logger = logproc.Logger(saved_dir=args.log_saved_dir, 
                                 title_name=args.log_title_name)
    
    
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        main_logger.write("main", "__main__", "Info", "Using gpu index is " + str(args.gpu_index) + ".")
        
        local_args = loadcfg4ds(json_cfg_file=args.config_path)

        local_args["kernel"]["log"] = { "title_name": args.log_title_name, "saved_dir": args.log_saved_dir }  
        main_logger.write("main", "__main__", "Info", "DeepStream app is initializing.")
        app = streamer.DeepStreamApp(local_args=local_args, 
                                     gpu_index=args.gpu_index)
    except Exception as init_error:
        main_logger.write("main", "__main__", "Error", "The initialization has some errors: " + str(init_error))
        sys.exit(0)
    
    main_logger.write("main", "__main__", "Info", "DeepStream app is executing.")
    app.execute(rtsp_url=args.rtsp_urls)