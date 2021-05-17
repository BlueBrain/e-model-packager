# Copyright (c) BBP/EPFL 2018; All rights reserved.                         
# Do not distribute without further notice.   

./compile_mechanisms.sh

if [ $# -eq 0 ]
then
    python -m emodelrunner.run
else
    python -m emodelrunner.run --c $1
fi
