#!/bin/sh

set -e

INSTALL_DIR=$1
MOD_DIR=$2

cd ${INSTALL_DIR}
cp /gpfs/bbp.cscs.ch/project/proj32/ajaquier/mechanisms/GluSynapse.mod ${MOD_DIR}
 
echo "Building mod files"
rm -rf x86_64
nrnivmodl ${MOD_DIR}
cd -
