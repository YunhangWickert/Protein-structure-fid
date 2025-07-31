# Protein Structure FID Evaluation

This repository contains code for evaluating the Fréchet Inception Distance (FID) between protein structure embeddings generated from PDB files. The FID is a commonly used metric for assessing the similarity between two distributions, and in this case, it's applied to protein structures.

## Environment Setup

To set up the environment for running the code, follow these steps:

1. **Quickly Start**:
 ```sh
   git clone https://github.com/YunhangWickert/Protein-structure-fid.git
   cd Protein-structure-fid
   conda env create -f environment.yml
   conda activate pfmbench
   python compute_protein_fid.py --ref_pdb_directory /path/to/ref.pdb --pre_pdb_directory /path/to/pre.pdb 
```
## Parameters:
--ref_pdb_directory: Directory containing reference PDB files.
--pre_pdb_directory: Directory containing design PDB files.
