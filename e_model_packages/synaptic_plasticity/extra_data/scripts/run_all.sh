# Copyright (c) BBP/EPFL 2021.
# This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/) 

if [ ! -f "x86_64/special" ]; then
    nrnivmodl mechanisms
fi

if [ $# -eq 0 ]
then
    search_dir = "./config"
else
    search_dir=$1
fi

for config_path in "$search_dir"/*.ini
do
    python -m emodelrunner.run_synplas --config_path $config_path
done
