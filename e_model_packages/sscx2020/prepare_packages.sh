CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packages/sscx2020

LUIGI_CONFIG_PATH=e_model_packages/sscx2020/luigi.cfg luigi --module workflow CollectMEModels --local-scheduler
