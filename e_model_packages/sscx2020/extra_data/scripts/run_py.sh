# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

# preloaded=False not implemented yet in BluePyOpt mechanisms
if [ ! -f "x86_64/special" ]; then
    nrnivmodl mechanisms
fi

if [ $# -eq 0 ]
then
    python run.py
else
    python run.py --c $1
fi
