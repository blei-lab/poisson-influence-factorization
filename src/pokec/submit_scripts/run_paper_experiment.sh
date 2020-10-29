#!/bin/bash

python -m pokec.run_experiment \
--data_dir=${DIR} \
--out_dir=${OUT} \
--model=${MODEL} \
--variant=${VARIANT} \
--num_components=${NUM_COMPONENTS} \
--num_exog_components=${NUM_EXOG_COMPONENTS} \
--confounding_type=${CONF_TYPES} \
--configs=${CONFIGS} \
--seed=${SEED}
