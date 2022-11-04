#!/bin/bash

## Presets
#SBATCH --job-name=ring_cmaes

#SBATCH -p compute-hugemem
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks=11
#SBATCH --mem=10G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd /mmfs1/gscratch/labfairhall/davidgbe
source davidgbe-env/bin/activate
cd $CURRDIR

python ring_attractor_test.py

deactivate

## Exit
exit 0