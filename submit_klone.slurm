#!/bin/bash

## Presets
#SBATCH --job-name=plastic_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=20
#SBATCH --mem=50G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd /mmfs1/gscratch/labfairhall/davidgbe
source davidgbe-env/bin/activate
cd $CURRDIR

python CMAES_test.py

deactivate

## Exit
exit 0