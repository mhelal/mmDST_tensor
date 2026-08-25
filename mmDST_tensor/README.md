# mmDST Tensor: Mathematics of Arrays for Multiple Sequence Alignment with Tensor Decomposition

This repository extends the original mmDST work (Helal et al., 2012) with modern Python/Numpy implementations and tensor decomposition routes.

## Repository Structure
mmDST_tensor/
├── sequential/ # Original C sequential implementation
├── mslave/ # Original MPI master-slave implementation
├── p2p/ # Original MPI peer-to-peer implementation
├── publications/ # Original publications
├── data/ # Test and real datasets
├── python/ # Python implementations
│ ├── core/ # Core MoA implementations
│ ├── alignment/ # Global and local alignment
│ ├── tensor/ # Tensor decomposition routes
│ ├── utils/ # Utilities (Balibase, motif scanning)
│ └── benchmarks/ # Benchmark scripts
├── experiments/ # Experimental results
│ ├── scaling/ # k-scaling results
│ ├── ablation/ # Band sensitivity and timing
│ ├── validation/ # BAliBASE and JASPAR validation
│ └── figures/ # Generated figures
├── scripts/ # Helper scripts
├── results/ # All experimental results
└── docs/ # Documentation

text

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Conda (recommended)

### Setup Environment
```bash
conda env create -f environment.yml
conda activate msa_tensor
Download Datasets
bash
# BAliBASE
cd data
wget http://www.lbgi.fr/balibase/Balibase_R1-5.tar.gz
tar -xzf Balibase_R1-5.tar.gz

# JASPAR (via Python)
python -c "from pyjaspar import jaspardb; jdb = jaspardb(release='JASPAR2024'); motifs = jdb.fetch_motifs()"
Reproducing Results
Full Pipeline
bash
# Run all experiments
python experiments/run_all.py

# Generate figures
python experiments/figures/generate_figures.py
Individual Experiments
bash
# Scaling with k
python experiments/scaling/run_scaling.py

# Band sensitivity
python experiments/ablation/band_sensitivity.py

# BAliBASE validation
python experiments/validation/balibase_ablation.py

# Local alignment
python experiments/validation/local_ablation.py
Results
Key results from the experiments:

Compression: 5,220× at k=5 (TT decomposition)

Speed: 3× faster than exact with ε=1

Accuracy: 100% score preservation with 97.8% cell reduction

Full-length: 99.7% cell reduction on BAliBASE BB11001

Citation
If you use this code, please cite:

text
Helal, M., Mullin, L., Potter, J., & Sintchenko, V. (2009). 
Search space reduction technique for distributed multiple sequence alignment. 
In NPC 2009.
License
GPL v2 (see LICENSE file)

Contact
Original: mhelal@cse.unsw.edu.au
