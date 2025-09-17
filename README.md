# CDCT-AO: A Novel Curiosity-Driven Convolutional Transformer with Adaptive Optimization for Offline-to-Online Multi-Agent Reinforcement Learning (StarCraft II) 🌝🌝🌝

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)

Official implementation of our anonymous journal submission:  
**"CDCT-AO: A Novel Curiosity-Driven Convolutional Transformer with Adaptive Optimization for Offline-to-Online Multi-Agent Reinforcement Learning"**

[[Paper]](https://albert-jin.github.io/) | [[Project Page]](https://github.com/albert-jin/CDCT-AO) | [[SMAC Benchmark]](https://github.com/oxwhirl/smac)

<!-- https://arxiv.org/abs/XXXX.XXXXX -->
---

## 🚀 Key Features
- **MA-CCT**: Multi-Agent Causal-convolutional Transformer integrating causal convolution filters into Transformer architecture
- **CurExp**: Novel curiosity-driven exploration mechanism using next-state prediction error
- **MSAL**: Multi-Stage Adaptive Learning-rate scheduler for robust optimization
- **State-of-the-art performance** on StarCraft II Micromanagement (SMAC) benchmark
- **Enhanced OOD Generalization**: Superior handling of out-of-distribution states through improved exploration

---
- Main Architecture of CDCT-AO
<p align="center">
  <img src="./source/CDCTAO-Overall.png" width="750" alt="CDCT-AO Architecture">
</p>

---
- Structure of MA-CCT
<p align="center">
  <img src="./source/MA-CCT.png" width="750" alt="MA-CCT Structure">
</p>

---
- Structure of CurExp, Curiosity-driven Exploration Mechanism
<p align="center">
  <img src="./source/CurExp.png" width="750" alt="CurExp Exploration">
</p>

---
- Details of Multi-Stage Adaptive Learning-rate scheduler (MSAL)
<p align="center">
  <img src="./source/MSAL.png" width="750" alt="MSAL Details Table">
</p>

---
## 🧩 Key Components
- **MA-CCT Architecture**: 
  - Replaces MHSA in Transformer with causal-convolution based token-mixer
  - Better captures localized temporal patterns in MDP sequences
  - Enhanced generalization for OOD states through improved Markov property modeling
- **CurExp Mechanism**: 
  - Utilizes forward dynamics model for next-state transition prediction
  - Uses prediction error to guide exploration towards uncertain states
  - Bridges offline-online distribution gap during fine-tuning
- **MSAL Scheduler**: 
  - Multi-stage adaptive learning rate adjustment
  - Warmup phase for stability, exploration phase for adaptation, fine-tuning phase for convergence
  - Dynamic optimization based on real-time performance feedback

---
## 📌 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/albert-jin/CDCT-AO.git
cd CDCT-AO
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
bash install_sc2.sh
```

## 🏃 Quick Start

Run offline pre-training + online fine-tuning (offline-to-online) on 3s5z map:
```bash
python run_madt_sc2.py --map_name 3s5z --cuda_id 0
```

## 🏆 Performance

- Performance Comparisons of Online Fine-tuning
---
<p align="center">
  <img src="./source/main-finetune.png" width="800" alt="CDCT-AO Online Performance">
</p>

- Offline Comparisons Between Baselines and Our CDCT-AO
---
<p align="center">
  <img src="./source/main-offline.png" width="650" alt="CDCT-AO Offline Performance">
</p>

## 🏆🏆 Ablations

- Ablation of CurExp comparison in CDCT-AO ➡️ (Curiosity-Driven Exploration vs. Without Curiosity-Driven Exploration)
---
<p align="center">
  <img src="./source/CurisAbla.png" width="800" alt="CDCT-AO ablated Performance 1">
</p>

- Ablation of MA-CCT in CDCT-AO ➡️ (Causal-Convolution vs. Transformer MHSA)
---
<p align="center">
  <img src="./source/ConvAbla.png" width="800" alt="CDCT-AO ablated Performance 2">
</p>

- Ablation of MSAL in CDCT-AO ➡️ (Self-Adaptive MSAL with Other Learning Rate Schedulers)
---
<p align="center">
  <img src="./source/LRAbla.png" width="800" alt="CDCT-AO ablated Performance 3">
</p>

## 🛠️ Customization
Modify `./models/gpt_model.py` to:
- Adjust causal-convolution architectures 
- Experiment with different token mixing strategies 
- Tune convolution filter parameters

Modify `./models/curiosity_model.py` to:
- Implement custom curiosity mechanisms 
- Modify prediction error calculation 
- Adjust exploration bonus weights

Edit `./framework/utils.py` & `./framework/trainer.py` & `./framework/rollout.py` to:
- Customize learning rate schedules 
- Modify phase transition thresholds 
- Implement alternative adaptive strategies

---

## 🤝 Contributing
We welcome contributions! Please open an issue or submit PRs for:
- ~~New SMAC scenarios~~
- Alternative policy coordination mechanisms
- Performance optimizations

This project is forked from [ReinholdM/Offline-Pre-trained-Multi-Agent-Decision-Transformer](https://github.com/ReinholdM/Offline-Pre-trained-Multi-Agent-Decision-Transformer), many thanks to their foundational research work.

## 📜 Citation
If you use this work, please cite:
```bibtex
@article{shi2025cdctao,
  title={CDCT-AO: A Novel Curiosity-Driven Convolutional Transformer with Adaptive Optimization for Offline-to-Online Multi-Agent Reinforcement Learning},
  author={Shi, Bohang (Equal. Contributed.) and Jin, Weiqiang (Equal. Contributed.) and Zhao, Biao},
  journal={anonymous journal (under review)},
  year={2025},
}
