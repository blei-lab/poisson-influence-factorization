#!/bin/bash
#SBATCH -A sml
#SBATCH --mem-per-cpu=64gb
source activate influence
python -m pokec.run_sensitivity_study \
--data_dir=${DIR} \
--out_dir=${OUT} \
--num_components=${NUM_COMPONENTS} \
--num_exog_components=${NUM_EXOG_COMPONENTS} \
--seed=${SEED}
