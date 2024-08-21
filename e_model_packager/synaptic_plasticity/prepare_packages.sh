CURRENT_DIR=$PWD
export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packager/synaptic_plasticity

MY_PATH=/gpfs/bbp.cscs.ch/project/proj32/ajaquier/myenv37-glusyn

BGLIBPY_MOD_LIBRARY_PATH=$MY_PATH/.neurodamus/local/x86_64/.libs/libnrnmech.so
export BGLIBPY_MOD_LIBRARY_PATH
HOC_LIBRARY_PATH=$MY_PATH/.neurodamus/local/neurodamus-core/hoc
export HOC_LIBRARY_PATH

rm -rf output/L23PC_L23PC/16903-11518/

LUIGI_CONFIG_PATH=e_model_packager/synaptic_plasticity/luigi.cfg luigi --module workflow RunWorkflow --local-scheduler
