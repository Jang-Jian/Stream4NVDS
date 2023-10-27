import os
import datetime


class Logger(object):
    """
    log processing
    """
    def __init__(self, saved_dir: str = "./log", title_name: str = "ds") -> None:
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        self.__saved_log = os.path.join(os.path.abspath(saved_dir), self.__getLogFile(title_name))

    def __getLogFile(self, title_name: str = "") -> str:
        """
        initialize the log file.
        """
        valstr = lambda value: str(value) if value > 9 else "0" + str(value)
        now_time = datetime.datetime.now()
        yyyy_str = valstr(now_time.year)
        mm_str = valstr(now_time.month)
        day_str = valstr(now_time.day)
        return title_name + "_" + yyyy_str + mm_str + day_str + ".log"

    def write(self, obj_name: str, func_name: str, status_name: str, information: str):
        """
        Example. \n
        obj_name: "logproc" \n
        func_name: "Logger.write" \n
        status_name: "Info" \n
        information: "writing log." \n

        p.s status_name supports 'Info', 'Error', 'Warn', 'Debug' & 'End'.
        """
        obj_name_str = ""
        if obj_name is not None:
           obj_name_str = "[" + str(obj_name) + "]"
        
        func_name_str = ""
        if func_name is not None:
            func_name_str = "[" + str(func_name) + "]"

        status_str = ""
        if status_name == "Info":
            status_str = "[Info]"
        elif status_name == "Error":
            status_str = "[Error]"
        elif status_name == "Warn":
            status_str = "[Warn]"
        elif status_name == "Debug":
            status_str = "[Debug]"
        elif status_name == "End":
            status_str = "[End]"

        happened_time = "[" + datetime.datetime.now().strftime('%Y-%m-%d %T') + "]"
        full_log_str = happened_time + obj_name_str + func_name_str + status_str + ": " + information

        print(full_log_str)

        with open(self.__saved_log, 'a') as file:
            file.write(full_log_str + "\n")