# FAST: Federated Active Learning with Foundation Models for Communication-efficient Sampling and Training

<div align="center">

<div>
    <a href='https://www.linkedin.com/in/haoyuan-li-cs9654/' target='_blank'>Haoyuan Li</a><sup>1</sup>&emsp;
    <a href='https://mathias-funk.com/' target='_blank'>Mathias Funk</a><sup>1</sup>&emsp;
    <a href='https://jd92.wang/' target='_blank'>Jindong Wang</a><sup>2</sup>&emsp;
    <a href='https://aqibsaeed.github.io/' target='_blank'>Aaqib Saeed</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup>Eindhoven University of Technology&emsp;
<sup>2</sup>College of William & Mary&emsp;
</div>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2504.03783-blue?logo=arxiv&logoColor=orange)](https://arxiv.org/abs/2504.03783)
[![Project Page](https://img.shields.io/badge/Project%20Page-Online-brightgreen)](https://haoyuan-l.github.io/fast/)

</div>

## 📝 Abstract

FAST is a two-pass federated active learning framework that reduces communication costs by **8x** while achieving **4.36% higher accuracy** than existing methods. We leverage foundation models for weak labeling, followed by human refinement of uncertain samples, using only **5% labeling budget**.

## 🌟 Key Features

- **Two-Pass Strategy**: Foundation model weak labeling + human refinement
- **8x Communication Reduction**: Fewer rounds, lower costs
- **Superior Performance**: 4.36% average improvement over baselines
- **Minimal Annotation**: Only 5% labeling budget required

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision numpy flwr transformers
git clone https://github.com/Haoyuan-L/FAST-official.git
cd FAST-official
```

### Basic Usage
```bash
# CIFAR-10 with default settings
python main.py --dataset cifar10 --num_clients 10 --budget 0.05

# Medical datasets
python main.py --dataset pathmnist --foundation_model evaclip

# Compare with baselines
python run_baselines.py --methods random,entropy,kafal
```

## 📊 Results

| Method | CIFAR-10 | SVHN | PathMNIST | Budget | Rounds |
|--------|----------|------|-----------|---------|--------|
| Random | 64.19 | 80.90 | 68.41 | 20% | 400 |
| KAFAL | - | - | - | 20% | 400 |
| **FAST** | **77.14** | **87.91** | **88.48** | **5%** | **100** |

**Communication Efficiency**: 87.3% cost reduction, 76.7% time reduction

## 🔧 Configuration

Key parameters:
- `--dataset`: [cifar10, cifar100, svhn, pathmnist, dermamnist]
- `--foundation_model`: [siglip, clip, evaclip, dinov2]
- `--budget`: Labeling budget (default: 0.05)
- `--num_clients`: Number of clients (default: 10)

## 📜 Citation

```bibtex
@article{li2025fast,
  title={FAST: Federated Active Learning With Foundation Models for Communication-Efficient Sampling and Training},
  author={Li, Haoyuan and Funk, Mathias and Wang, Jindong and Saeed, Aaqib},
  journal={IEEE Internet of Things Journal},
  year={2025},
  publisher={IEEE}
}
```

## 📧 Contact

**Haoyuan Li**: [h.y.li@tue.nl](mailto:h.y.li@tue.nl) | [Issues](https://github.com/Haoyuan-L/FAST-official/issues)
