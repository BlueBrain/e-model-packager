CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packages/sscx2020

LUIGI_CONFIG_PATH=e_model_packages/sscx2020/luigi_release.cfg luigi --module workflow ParseCircuit --mtype=L5_TPC:A --etype=cADpyr --region=S1ULp --gidx=79598 --local-scheduler
