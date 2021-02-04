#!/bin/bash
BASE_DIR=/proj/sml/projects/social-influence
export DIR=/proj/sml/projects/social-influence/pokec/regional_subset/
export NUM_COMPONENTS=5
export NUM_EXOG_COMPONENTS=5

for SIM_ITER in {1..10};
do
	export SEED=${SIM_ITER}
	export OUT=${BASE_DIR}/pokec_sensitivity_study/${SIM_ITER}/
	sbatch --job-name=pokec_sensitivity_analysis_${SIM_ITER} --output=${BASE_DIR}/out/pokec_sensitivity_analysis_${SIM_ITER}.out pokec/submit_scripts/cluster_jobs/run_sensitivity_experiment.sh
done
