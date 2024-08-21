MTYPE=L5_TPC:A
ETYPE=cADpyr
GIDX=79598
GID=4138379

rm -rf output/memodel_dirs/$MTYPE/$ETYPE/$MTYPE_$ETYPE_$GIDX/
rm output/luigi-profiling.txt
rm output/metype_gids.json
rm output/system-state.log

CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packager/sscx2020

# Those Tasks have to be run one after the other to ensure that one Task will not affect the memory / disk space properties of another.
# This is why no WrapperTask was used.
# RunHoc by construction rerun CreateHoc, so RunHoc benchmark data might be biased.
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow PrepareOutputDirectory  --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow CreateSystemLog --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow PrepareMEModelDirectory --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow CreateHoc --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow RunHoc --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow RunPyScript --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow RunOldPyScript --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow DoRecordings --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow CreateMETypeJson --mtype=$MTYPE --etype=$ETYPE --gid=$GID --gidx=$GIDX --local-scheduler
LUIGI_CONFIG_PATH=e_model_packager/sscx2020/luigi.cfg luigi --module workflow ParseCircuit --mtype=$MTYPE --etype=$ETYPE --gidx=$GIDX --local-scheduler
