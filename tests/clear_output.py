import os
import shutil

path_to_output = os.path.join("e_model_packages", "sscx2020", "output")

if os.path.isdir(path_to_output):
    shutil.rmtree(path_to_output)
os.makedirs(path_to_output)
