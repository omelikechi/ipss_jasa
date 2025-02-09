# Supplementary Code for "Integrated Path Stability Selection," submitted to JASA

`main.py` implements Integrated Path Stability Selection (IPSS) as well as stability selection and cross-validation methods used for comparison.  
`helpers.py` contains helper functions for `main.py`.  
`base_selectors.py` contains the base feature selectors used with IPSS and stability selection.

The `simulations` folder contains code for running the simulations in Section 5 of the manuscript. The main file here is `run_simulation.py`, which can be adjusted to reproduce all results in Section 5 and Sections S7 and S8 of the Supplement. 
`generate_data.py` contains code for generating synthetic datasets used in the simulation experiments.

The `applications` folder contains code for reproducing the real data results in Section 6.  
- `prostate.py` applies the methods to the prostate cancer dataset.  
- `colon.py` applies the methods to the colon cancer dataset.

### **Available Datasets**
- **RNA-sequencing data** from ovarian cancer patients: `simulations/ovarian_rnaseq.npy`
- **Protein data** from prostate cancer patients: `applications/prostate_data.npy`
- **Colon cancer dataset**: `applications/colon_data.npy`
