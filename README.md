**Introduction**

This repository contains the code and demo data for: 
- [1] _Estimating Social Influence from Observational Data_, by Dhanya Sridhar, Caterina De Bacco and David Blei.

If you use this code please cite [1].   
The paper can be found (ADD LINK) here (preprint).  
 
Social influence the causal inference problem of determining how users affect the behavior of their friends in a network.
For example, if a user shares a news article today and tomorrow, her friend does the same, we want to know if the 
article sharing behavior of the friend was caused by the user.
The paper introduces the Poisson Influence Factorization (PIF) algorithm for estimating social influence from observed network and behavior data.
The algorithm uses the observed data to estimate the latent preferences people have for behaviors, and for forming social ties. These latent preferences
are confounders: they may explain much of the shared behavior that friends exhibit. Put simply, the two users who shared articles may have done so
simply because they liked the article, and the hidden reasons they like the article also caused them to be friends in the first place.
preferences for different 
The PIF algorithm uses probabilistic factor models to estimates latent variables from network and behavior data, and uses them in a causal adjustment procedure
to estimate influence.
The demo data involves a real social network called Pokec, and simulated item purchases.

**Requirements** 

The code has been tested on Python 3.6.9 with the following packages:
```bash
conda==4.7.12
numpy==1.16.4
pandas==0.25.0
scikit-learn==0.21.2
scipy==1.3.0
```

It is possible to install the dependencies using pip:
```bash
pip install -r requirements.txt
```

**Instructions**

To run the code with a particular choice of parameters, type on a terminal inside the directory ```/src```:
```bash
python -m pokec.run_experiment
``` 
The default output file will be ```./out/conf=(50, 50);conf_type=both.npz``` (with the default values of the parameters).  To load the results, use for instance the python commands:
```bash
import numpy as np 
infile = './out/conf=(50, 50);conf_type=both.npz'
res = np.load(infile)
```
The fitted results can be accessed using  ```res['fitted']```, the ground truth with  ```res['true']```.  
You can change the default parameters by passing them via command line with the corresponding flag. See directly the code ```pokec/run_experiment.py``` for a list of flags and their default values.  	

Alternatively, you can run a script to reproduce the results from the semi-simulated experiments (Table 3) in the paper (and include the additional PIF-variant studied in the appendix). This will run the code for all set of parameters presented in the paper.  
To use the code for this purpose, run all scripts from ```/src```.

1. Run the command
```./pokec/submit_scripts/submit_paper_experiment.sh```
The outputs will be written to ```../out/pokec_paper_results```.  
Running the whole set of experiments to replicate the paper's results can take approximately 3h-5h on a standard laptop, it can be much faster if launched in parallel (e.g. on a computational cluster). You can customize the range of parameters by directly modifying the executable ```./pokec/submit_scripts/submit_paper_experiment.sh```.

2. Use the command ```jupyter notebook``` and open ```./src/pokec/pokec_paper_results.ipynb``` and play the entire notebook. Table 3 in the paper (and the extra result from the appendix) should be reproduced exactly.
	
