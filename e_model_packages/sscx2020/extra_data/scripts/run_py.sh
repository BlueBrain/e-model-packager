# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

if [ $# -eq 0 ]
then
    source config/config.ini
else
    source config/$1
fi

MY_PATH=memodel_dirs/$mtype/$etype/${mtype}_${etype}_${gidx}
# preloaded=False not implemented yet in BluePyOpt mechanisms
if [ ! -f "x86_64/special" ]; then
    nrnivmodl $MY_PATH/mechanisms
fi

if [ $# -eq 0 ]
then
    python run.py
else
    python run.py --c $1
fi
