# **Supplementary code for "Integrated path stability selection" (JASA submission)**

The arXiv version of the paper: [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

## **Important!**  
This repository contains code for reproducing the results and figures in the paper *"Integrated Path Stability Selection,"* submitted to JASA.  

The full **integrated path stability selection (IPSS) package**, which includes expanded functionality and is designed for general use, is 
available here: [https://github.com/omelikechi/ipss](https://github.com/omelikechi/ipss)

## **Description**
- `main.py` implements IPSS as well as stability selection and cross-validation methods used for comparison 
- `helpers.py` contains helper functions for `main.py`
- `base_selectors.py` contains the base estimators used with IPSS and stability selection
- The `simulations` folder contains
	- `figure1.py`: Reproduces Figure 1 in the paper
	- `figure2.py`: Reproduces Figure 2 in the paper
	- `generate_data.py`: Simulates data (Section 5)
	- `run_simulation.py`: Runs simulation experiments (Sections 5, S7, and S8)
	- `simulation_function.py`: The main simulation function
- The `applications` folder contains
	- `prostate.py`: Reproduces prostate cancer results (Section 6.1)
	- `colon.py`: Reproduces colon cancer results (Section 6.2)

## **Available Datasets**
This repository includes processed datasets used in the paper. The original sources are linked below:
- RNA-sequencing (RNAseq) data from ovarian cancer patients
	- Source: [https://www.linkedomics.org/data_download/TCGA-OV/](https://www.linkedomics.org/data_download/TCGA-OV/)
	- Processed file: `simulations/ovarian_rnaseq.npy`
- Reverse phase protein array (RPPA) data from prostate cancer patients 
	- Source: [https://www.linkedomics.org/data_download/TCGA-PRAD/](https://www.linkedomics.org/data_download/TCGA-PRAD/) 
	- Processed file: `applications/prostate_data.npy`
- Colon cancer data:
	- Source: [http://genomics-pubs.princeton.edu/oncology/affydata/index.html](http://genomics-pubs.princeton.edu/oncology/affydata/index.html) 
	- Processed file: `applications/colon_data.npy`

## **Additional notes**
- Additional details about the code, including approximate runtime, are provided at the top of each script
- Data generation involves randomness, so results may differ slightly from those in the paper
- In the paper, regularization parameters are denoted by **λ (lambda)**, and the probability measure parameter is denoted by **α (alpha)**. However, since `lambda` is a reserved keyword in Python, the code instead uses:
  - **α (alpha) → δ (delta)** for the probability measure parameter 
  - **λ (lambda) → α (alpha)** for regularization parameters 




