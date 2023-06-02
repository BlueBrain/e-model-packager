#!/bin/sh

set -e

MOD_DIR=$1
 
echo "Building mod files"
rm -rf x86_64
nrnivmodl ${MOD_DIR}
