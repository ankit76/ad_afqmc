import os

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
