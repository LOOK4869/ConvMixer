[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WG_g337P)
# ConvMixer Reproduction Project

## ECBM E4040 Deep Learning and Neural Networks - Fall 2025

---

## Project Overview

This project reproduces the results from:
> Asher Trockman, J. Zico Kolter  
> Carnegie Mellon University and Bosch Center for AI  
> arXiv:2201.09792

### Core Hypothesis

The paper argues that the strong performance of Vision Transformers may be primarily due to the **patch-based input representation** rather than the Transformer architecture itself.

### Our Contributions

1. Implemented ConvMixer architecture in PyTorch
2. Reproduced baseline results on CIFAR-10
3. Conducted kernel size ablation study
4. Conducted patch size ablation study
5. Analyzed and compared results with the original paper

---

## Notebook
### 00_data_check.ipynb
Verifies the CIFAR-10 data loading pipeline, visualizes sample images, and confirms the dataset configuration (50,000 training / 10,000 test samples, 10 classes). Ensures data preprocessing and DataLoader are working correctly before training.
### 01_Baseline CNN Training.ipynb
Establishes a performance baseline using a simple 3-layer CNN (~620K parameters). Achieves ~79% test accuracy on CIFAR-10, validating the training framework and providing a reference point for ConvMixer comparison.
### 02_convmixer_cifar10.ipynb
Implements and tests the ConvMixer architecture with a lightweight configuration (ConvMixer-256/8). Validates the patch embedding, depthwise/pointwise convolution blocks, and classification head are functioning correctly.
### 03_hyperparameter_tuning.ipynb
Explores different learning rates and training configurations to optimize ConvMixer performance. Identifies optimal settings (lr=0.001, AdamW with weight decay, gradient clipping) achieving 84% accuracy.
### 04_ablation_study.ipynb
Reproduces key ablation experiments from the paper: (1) Kernel size ablation (k=3,5,7,9) confirms larger kernels improve spatial mixing; (2) Patch size ablation (p=1,2,4) validates that smaller patches preserve more spatial information. Results are compared against the original paper's findings.
### main_convmixer_reproduction.ipynb
The comprehensive notebook that integrates the complete ConvMixer reproduction workflow. Includes model architecture visualization, baseline training (ConvMixer-256/8 achieving ~89% accuracy), kernel size ablation, and patch size ablation experiments. This file mainly presents the specific results of the complete framework "kernel size ablation" and "patch size ablation" experiments in 04_ablation_study.ipynb

---

## Results

### Baseline Reproduction

| Model | Configuration | Our Accuracy | Paper Accuracy |
|-------|--------------|--------------|----------------|
| ConvMixer-256/8 | k=9, p=1 | 88.98% | 95.88% |

### Kernel Size Ablation

| Kernel Size | Our Accuracy | Paper Accuracy | Δ |
|-------------|--------------|----------------|---|
| k=3 | 85.65% | 93.61% | -7.96% |
| k=5 | 86.98% | 95.19% | -8.21% |
| k=7 | 88.40% | 95.80% | -7.40% |
| k=9 | 88.94% | 95.88% | -6.94% |

**Finding**: Larger kernel sizes improve accuracy, supporting the importance of large receptive fields for spatial mixing.

### Patch Size Ablation

| Patch Size | Internal Resolution | Our Accuracy | Paper Accuracy |
|------------|---------------------|--------------|----------------|
| p=1 | 32×32 | 89.97% | 95.88% |
| p=2 | 16×16 | 86.03% | 95.00% |
| p=4 | 8×8 | 80.96% | 92.61% |

**Finding**: Smaller patches preserve more spatial information and achieve higher accuracy.

---

## Key Insights
We observe that our reproduced accuracy is slightly lower than the reported value. A primary factor is limited training duration: due to resource and environment constraints, we trained for only 30 epochs instead of the 200 epochs used in the original study. On a single GCP GPU, 30 epochs already required roughly 10 hours per ablation configuration, and extending to 200 epochs would have multiplied the compute cost substantially. Additionally, long-running jobs repeatedly suffered unexpected network or session disconnects, making completion of full-length training impractical within our project timeline. 

Beyond training duration, other potential sources of discrepancy include differences in augmentation strength, learning-rate scheduling, random seed effects, and minor implementation details (e.g., normalization layers or weight initialization). A systematic hyperparameter sweep, longer training, and multi-seed evaluation could further reduce variance and improve comparability with the original results.

Our reproduction study provides empirical evidence that a purely convolutional model equipped with patch embeddings can achieve strong performance on CIFAR-10 without employing self-attention. This supports the claim that patch-based representations and isotropic designs are key contributors to the success of modern vision architectures, and that attention is not strictly necessary to obtain competitive results in small-scale image classification.

The kernel size ablation indicates that increasing depthwise kernel size is an effective mechanism for expanding the receptive field and improving spatial mixing. While depthwise separable convolutions keep parameter counts manageable, large kernels are computationally more expensive, and we observe a noticeable reduction in training throughput for $k=9$ compared to $k=3$.

The patch size ablation highlights a clear accuracy--efficiency trade-off. Smaller patches preserve more spatial detail and achieve better accuracy, but increase computational cost due to higher internal resolution and larger activation tensors. In practice, intermediate patch sizes may represent a reasonable compromise under tight memory or latency constraints.

This study has several limitations. First, experiments are restricted to CIFAR-10 and a narrow range of ConvMixer configurations. Second, due to compute constraints, extensive hyperparameter sweeps and multi-seed evaluations are not performed. Third, our pipeline may differ slightly from the original implementation in augmentation parameters and scheduling. Future work could extend evaluation to larger datasets (e.g., ImageNet-1k), explore broader architectural settings (varying $h$ and $d$), and investigate efficiency-optimized implementations of large-kernel depthwise convolutions.

---

## Team Contributions

| Team Member | Contributions | Percentage |
|-------------|---------------|------------|
| Zehao Li | initialization; ConvMixer implementation; README and documentation | 35% |
| Minghao Liu | Training pipeline; data augmentation; main training experiments | 30% |
| Ke Lu | Ablation studies; result visualization; notebook cleanup and final integration | 35% |

---

## References

1. A. Trockman and J. Z. Kolter, “Patches are all you need?,” arXiv preprint, arXiv:2201.09792, 2022.

---

## License

This project is for educational purposes as part of ECBM E4040 at Columbia University.

## Project Structure

```
e4040-2025Fall-Project-LUCK-kl3753-zl3667-ml5312
.
├── configs
│   ├── __init__.py
│   └── config.py
├── E4040_2025Fall_LUCK_report_kl3753_zl3667_ml5312.pdf
├── figures
│   ├── architecture.png
│   ├── baseline_cnn_training.png
│   ├── baseline_training.png
│   ├── convmixer_quick_train.png
│   ├── gcp_work_screenshot
│   │   ├── kl3753_gcp_work_example_screenshot_1.png
│   │   ├── kl3753_gcp_work_example_screenshot_2.png
│   │   ├── kl3753_gcp_work_example_screenshot_3.png
│   │   ├── ml5312_gcp_work_example_screenshot_1.png
│   │   ├── ml5312_gcp_work_example_screenshot_2.png
│   │   ├── ml5312_gcp_work_example_screenshot_3.png
│   │   ├── zl3667_gcp_work_example_screenshot_1.png
│   │   ├── zl3667_gcp_work_example_screenshot_2.png
│   │   └── zl3667_gcp_work_example_screenshot_3.png
│   ├── kernel_size_ablation_cpu.png
│   ├── kernel_size_ablation.png
│   ├── lr_comparison.png
│   ├── patch_size_ablation_cpu.png
│   ├── patch_size_ablation.png
│   └── training_curves_tuned.png
├── logs
│   └── ablation_results.json
├── notebook
│   ├── 00_data_check.ipynb
│   ├── 01_Baseline CNN Training.ipynb
│   ├── 02_convmixer_cifar10.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_ablation_study.ipynb
│   └── main_convmixer_reproduction.ipynb
├── README.md
├── src
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── convmixer_block.py
│   │   ├── convmixer.py
│   │   └── patch_embedding.py
│   ├── train_utils.py
│   └── utils
│       ├── __init__.py
│       ├── augmentation.py
│       └── visualization.py
└── train.py

9 directories, 40 files

```

