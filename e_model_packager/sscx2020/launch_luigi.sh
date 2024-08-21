MTYPE='L6_BP'
ETYPE='dSTUT'
GID=142161
GIDX=5

CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packager/sscx2020
# luigi --module e_model_packager.sscx2020.workflow PrepareMEModelDirectory --local-scheduler --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX
#luigi --module workflow CompareVoltages --local-scheduler --mtype=$MTYPE --etype=$ETYPE --gidx=$GIDX

# luigi --module workflow ParseCircuit --local-scheduler

LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow PrepareMEModelDirectory --mtype=L23_BP --etype=bNAC --gidx=150 --gid=111728 --local-scheduler
