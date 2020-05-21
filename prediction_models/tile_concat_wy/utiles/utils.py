import os

def check_folder_exists(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            return False
        except:
            check_folder_exists(os.path.dirname(path))
            os.mkdir(path)
    else:
        return True