# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

if [ $# -eq 0 ]
then
    source config/config.ini
else
    source config/$1
fi

# preloaded=False not implemented yet in BluePyOpt mechanisms
if [ ! -f "x86_64/special" ]; then
    nrnivmodl mechanisms
fi

if [ $# -eq 0 ]
then
    python old_run.py
else
    python old_run.py --c $1
fi
