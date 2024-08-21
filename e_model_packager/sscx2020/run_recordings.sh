CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packager/sscx2020

LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow DoRecordings --mtype=L5_TPC:A --etype=cADpyr --gid=4138379 --region=S1ULp --gidx=79597 --configfile=config_synapses.ini --local-scheduler
