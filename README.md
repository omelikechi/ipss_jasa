# **Code for "Integrated path stability selection" (JASA)**

Paper (arXiv preprint): [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

## **Important!**  
This repository contains code for reproducing the results and figures in the paper *"Integrated Path Stability Selection,"* submitted to JASA.  

The full **integrated path stability selection (IPSS) package**, which includes expanded functionality and is designed for general use, is 
available here: [https://github.com/omelikechi/ipss](https://github.com/omelikechi/ipss)

## **Description**
- The `ipss` folder contains
	- `main.py` implements IPSS as well as stability selection and cross-validation methods used for comparison 
	- `helpers.py` contains helper functions for `main.py`
	- `base_selectors.py` contains the base estimators used with IPSS and stability selection
- The `simulations` folder contains
	- `figure1.py` reproduces Figure 1 in the paper
	- `figure2.py` reproduces Figure 2 in the paper
	- `generate_data.py` simulates data (Section 5)
	- `simulation.py` runs the simulation experiments (Sections 5, S7, and S8)
	- `simulation_function.py` is the main simulation function
- The `applications` folder contains
	- `prostate.py` reproduces the prostate cancer results (Section 6.1)
	- `colon.py` reproduces the colon cancer results (Section 6.2)
- The `data` folder contains the cleaned data used in this work
	- `colon_data.npy` is the colon cancer dataset (62-by-1908 numpy array)
	- `ovarian_data.npy` is the ovarian cancer dataset (569-by-6426 numpy array)
	- `prostate_data.npy` is the prostate cancer dataset (351-by-125 numpy array)

## **Available Datasets**
All three datasets used in this paper are provided as cleaned `.npy` files in the data folder. These are cleaned
versions of the original datasets, which can be accessed at the links below.
- Colon cancer data:
	- Source: [http://genomics-pubs.princeton.edu/oncology/affydata/index.html](http://genomics-pubs.princeton.edu/oncology/affydata/index.html) 
	- Detailed descriptions and links to the raw data are provided at the above link
- RNA-sequencing (RNAseq) data from ovarian cancer patients
	- Source: [https://www.linkedomics.org/data_download/TCGA-OV/](https://www.linkedomics.org/data_download/TCGA-OV/)
	- In the "OMICS Dataset" column, find the "RNAseq (GA, Gene level)" row (569 samples, 6426 attributes)
	- Click "Download" in that row to download the data (cct format)
- Reverse phase protein array (RPPA) data from prostate cancer patients 
	- Source: [https://www.linkedomics.org/data_download/TCGA-PRAD/](https://www.linkedomics.org/data_download/TCGA-PRAD/) 
	- In the "OMICS Dataset" column, find the "RPPA (Gene Level)" row (352 samples, 152 attributes)
	- Click "Download" in that row to download the data (cct format)

## **Additional notes**
- Additional details about the code, including approximate runtime, are provided at the top of each script
- Data generation involves randomness, so results may differ slightly from those in the paper
- In the paper, regularization parameters are denoted by **λ (lambda)**, and the probability measure parameter is denoted by **α (alpha)**. However, since `lambda` is a reserved keyword in Python, the code instead uses:
  - **α (alpha) → δ (delta)** for the probability measure parameter 
  - **λ (lambda) → α (alpha)** for regularization parameters 




