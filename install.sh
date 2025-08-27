#!/bin/bash

#1) Create and activate environment
ENVS=$(conda info --envs | awk '{print $1}' )
if [[ $ENVS = *"HumanSensing"* ]]; then
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate HumanSensing
else
   echo "Creating a new conda environment for HumanSensing project..."
   conda env create -f environment.yml
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate HumanSensing
   #python setup.py install
   #exit
fi;

