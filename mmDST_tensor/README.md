# Semiring-Dependent Tensor Compressibility and Sparse GPU Wavefronts for Multiple Sequence Alignment

Research implementation and experimental records accompanying **“Semiring-Dependent Tensor Compressibility and Sparse GPU Wavefronts for Multiple Sequence Alignment”** (ICCTA 2026). The study separates (i) semiring-dependent functional fidelity, (ii) sparse dynamic-programming (DP) geometry and GPU dependency evaluation, and (iii) graph-free streamed GPU Forward computation with Tensor Ring (TR) fitting.

**Scope.** This is a research benchmark of simultaneous multi-sequence DP, not a production biological multiple sequence aligner. E3 computes a partition-banded Forward field first and then fits a TR to sampled field observations; it does **not** execute the DP recurrence directly in compressed TR form. Max-plus optimal alignment and Forward/partition aggregation are different objectives.

## Experiments

| Phase | Question | Implementation and evidence |
|---|---|---|
| **E1 — Semiring and tensor controls** | Does low tensor reconstruction error preserve the alignment functional? | Dense controlled experiments compare max-plus and Forward representations, divergence, temperature, and rank. |
| **E2 — Sparse bands and predecessor GPU** | What is the accuracy/locality/resource trade-off when restricting DP states? | Partition-banded sparse graphs, host-constructed CSR dependencies, CPU reference and GPU kernels; archived BAliBASE/JASPAR studies. |
| **E3 — Streamed GPU Forward–TR** | Can the Forward field be generated and approximated without a dense DP tensor or global sparse dependency graph? | On-demand predecessor lookup, bounded wave history, wave fusion, reservoir sampling and post-hoc TR fitting. |

The published E3 archive records **432 graph configurations and 1,296 rank-level evaluations** (global/local, k=3–6). These are archived measurements, **not** results of the small smoke tests below.

## Requirements and installation

Python 3.11, Conda, and (for GPU experiments) an NVIDIA GPU with compatible driver and CUDA runtime. Tested by the author on **WSL2 / RTX 4080 Laptop GPU (58 SMs)** using **PyTorch 2.7.1+cu128**, **CuPy 14.2.0**, and **NumPy 2.4.6**. The frozen E3 campaign in the paper used a different RTX 4090 Laptop GPU (76 SMs); do not interpret new hardware timings as identical to the paper.

```bash
conda env create -f environment.yml
conda activate msa_tensor
# Install a compatible CUDA-enabled PyTorch build separately.
# Verified WSL2 configuration (not necessarily appropriate for every GPU):
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python scripts/check_environment.py
python scripts/check_repository.py
python scripts/test_restored_reference.py
```

For other hardware, select the appropriate PyTorch wheel at https://pytorch.org/get-started/locally/. The `environment.yml` intentionally does not pin the original experiment's unrecorded dependency versions. Run all commands below from the repository root.

## Reproduce the experiments

### E1 — Dense semiring/tensor analysis

```bash
python experiments/e1/bridge_probes.py
python experiments/e1/alignment_scaling.py
python experiments/e1/build_paper_assets.py
```

The first two commands have been run successfully on the author's current machine. Outputs are under `experiments/e1/results_alignment/` and `experiments/e1/paper_assets/`. These are controlled synthetic experiments and require no downloaded datasets.

### E2 — Sparse graph and GPU CCR experiments

Check the restored CPU reference against the accelerated graph builder:

```bash
python scripts/test_restored_reference.py
```

Run a small, independently named GPU smoke test:

```bash
python experiments/e2/ccr_scaling_v2.py \
  --datasets synthetic-global --k 3 --part-sizes 1 --eps 0 \
  --trials 1 --global-L 4 --block-sizes 128 \
  --gpu-repeats 1 --gpu-warmup 0 \
  --out experiments/e2/test_reader
```

The author has run this small configuration successfully on the RTX 4080 Laptop GPU, with score recovery and a checkpoint written. The full E2 sweep is considerably more expensive, especially when the host-CSR graph becomes large. Inspect `--help` and begin with small cases. Archived results are in `experiments/e2/results/`.

### E3 — Streamed Forward and Tensor Ring

Run a small GPU test **with the CPU/GPU reference self-test enabled**:

```bash
python experiments/e3/experiment3_gpu_streaming_tr.py \
  --datasets synthetic-global --k 3 --part-sizes 1 --eps 0 \
  --divergence 0 --ranks 2 --trials 1 --global-L 6 \
  --reservoir-size 10000 --epochs 3 \
  --out experiments/e3/test_reader
```

The author has verified the output **`[self-test] GPU geometry and Forward match CPU reference`** and a successful rank-2 result on the RTX 4080 Laptop GPU. This verifies a small global control; it does not establish bitwise equivalence of the complete frozen E3 campaign or local seeded configurations.

To run the full synthetic factorial protocol (resource intensive):

```bash
python experiments/e3/experiment3_gpu_streaming_tr.py \
  --datasets synthetic-global,synthetic-local --k 3,4,5,6 \
  --part-sizes 1,2,4 --eps 0,1 --q 3 \
  --divergence 0,0.2,0.4 --ranks 2,4,8 \
  --trials 3 --seed 12345 --global-L 12 --motif-len 9 --flank-len 12 \
  --beta 0.1 --block-size 256 --small-wave-threshold 1024 \
  --small-bundle-max-waves 16 --max-blocks-per-sm 8 \
  --reservoir-size 1000000 --hard-vram-gb 11.5 \
  --epochs 60 --batch-size 65536 --lr 0.002 \
  --val-fraction 0.1 --functional-weight 20 \
  --early-stop-patience 8 \
  --out experiments/e3/new_runs/full_factorial
```

Choose a **new** `--out` directory; do not overwrite `experiments/e3/results/`. The runner checkpoints completed configurations for resumption. Local-anchor and biological-input equivalence should be checked before claiming an exact historical reproduction.

### Regenerate E3 figures without a GPU

```bash
python experiments/e3/e3_plot_source.py
```

Reads archived `experiments/e3/results/e3_stream_raw.csv` and related wave/protocol files, and writes tables and plots under `experiments/e3/tables/` and `experiments/figures/`. The paper's compression-baseline panel is `experiments/figures/fig_e3_tr_compression_baselines.pdf`; it compares TR parameter storage with both hypothetical dense-field storage and actually generated sparse-field storage. The former includes the benefit of state sparsification and must not be described as TR-only compression.

## Biological data and provenance

**E1 and E3:** Controlled synthetic sequences are generated by the experiment scripts; the E3 archived protocol and raw results are included.

**E2 / BAliBASE:** Obtain the relevant benchmark from the [BAliBASE project](https://www.lbgi.fr/balibase/). The paper reports 20 tractable RV11 global controls; the exact original sequence-selection manifest and input files are not bundled. Archived aggregate results are included but cannot, by themselves, establish a fresh reproduction of the same biological subset.

**E2 / JASPAR:** Obtain motif profiles from [JASPAR](https://jaspar.elixir.no/), using the appropriate release and organism selection. The original scanner input sequences, exact motif-selection manifest, and all preprocessing inputs are not bundled. The archived scanner and motif results are retained as evidence, not as a replacement for the missing source data.

The historical `scripts/download_data.sh` is **not a verified data installer**; it may generate synthetic sequences if a biological input is absent. Do not use generated sequences as biological benchmark data or report them as such.

## Repository structure

```text
python/                    Core, alignment, tensor and mmrepro CPU/GPU modules
experiments/e1/            Dense semiring/tensor controls
experiments/e2/            Host-CSR GPU, CCR sweep and archived biological results
experiments/e3/            Graph-free Forward–TR runner, archived E3 CSVs and plots
experiments/figures/       Generated paper and supplementary figures
scripts/                   Environment and CPU-reference checks
docs/                      WSL/GPU troubleshooting
```

## Interpretation and limitations

- Band restriction changes admissible alignment paths. E3 exact streamed controls are restricted to tractable state counts; higher-dimensional experiments use controlled synthetic references rather than unrestricted exact MSA.
- Unconstrained TR fitting does not guarantee positivity of every reconstructed Forward-field entry. A finite reconstructed functional is not a proof of global domain preservation.
- The archived E3 rank-level outcomes are stochastic and depend on optimisation, device and software versions. Published timings refer to the original GPU and should not be extrapolated to different hardware.
- The recovered CPU-reference, seed and dataset modules passed a small slow/fast sparse-graph regression test. The dataset reader includes a compatibility addition; this is not evidence of identical historical biological preprocessing.
- This repository preserves older auxiliary scripts and archived measurements. Only the entry points documented above have been specifically checked in the current restoration workflow.

## Citation

If this repository supports your work, cite the accompanying ICCTA 2026 paper **“Semiring-Dependent Tensor Compressibility and Sparse GPU Wavefronts for Multiple Sequence Alignment”**, by Manal Helal. Add the final proceedings DOI and bibliographic record when available.

## Contact

Manal Helal · School of Physics, Engineering and Computer Science, University of Hertfordshire · m.helal@herts.ac.uk
