# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

source config/config.ini
MY_PATH=memodel_dirs/$mtype/$etype/${mtype}_${etype}_${gidx}
# preloaded=False not implemented yet in BluePyOpt mechanisms
if [ ! -f "x86_64/special" ]; then
    nrnivmodl $MY_PATH/mechanisms
fi
python old_run.py
