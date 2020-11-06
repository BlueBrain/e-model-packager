CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packages/sscx2020

LUIGI_CONFIG_PATH=e_model_packages/sscx2020/luigi.cfg luigi --module workflow DoRecordings --mtype=L23_BP --etype=bNAC --gid=111728 --gidx=150 --configfile=config_synapses.ini --local-scheduler
