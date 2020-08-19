# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

source config/config.ini
MY_PATH=memodel_dirs/$mtype/$etype/${mtype}_${etype}_${gidx}
# preloaded=False not implemented yet in BluePyOpt mechanisms
nrnivmodl $MY_PATH/mechanisms
python old_run.py
