> *"Tiny consistent steps compound into transformations that once seemed impossible."*

I'm a researcher at **MILA / ÉTS Montréal** working on **mechanistic interpretability** — how to locate a computation inside a language model, characterise it, and edit it without breaking everything else — alongside **learning theory** (implicit bias, spurious correlations, generalization).

---
### 🌐 Connect with Me

[![Email](https://img.shields.io/badge/Email-grey?logo=gmail)](mailto:lompoaser9@gmail.com)
[![Website](https://img.shields.io/badge/Website-0A66C2?logo=googlechrome&logoColor=white)](https://aser97.github.io/Blog/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/AserLompo)
[![X (Twitter)](https://img.shields.io/badge/Twitter-black?logo=x)](https://x.com/aserlompo)
[![GitHub followers](https://img.shields.io/github/followers/Aser97?label=Follow&style=social)](https://github.com/Aser97)

---

### 🔬 Research

**[Making Learned Programs Explicit — mechanistic interpretability of arithmetic circuits](https://aser97.github.io/Blog/Projects/N2P/)** · *ongoing*
Circuit-level interpretability localises behaviour to **nodes** (heads, MLPs), but nodes are heavily shared across tasks — so any edit risks collateral damage. This project tests whether the right unit is the **direction**: whether number-value features live in a low-dimensional subspace that is far more task-**exclusive** than the components carrying it, making direction-level exclusivity a *certifiable safety criterion for model editing*.
Pipeline on **GPT-J** and **Llama-3-8B**: circuit identification via **Edge Pruning**, validated against **Tracr** programs with known ground-truth circuits, then dual-site **SAE** feature tracking, then inference-time substitution of the learned sub-computation by an exact external module.
Status: representation-level groundwork done across six operations and three prompt framings; causal validation is the next gate. → [code](https://github.com/galaxy532/N2P-Experiments)

**[Implicit Bias of Gradient Descent under Feature-Mediated Spurious Correlations](https://aser97.github.io/Blog/Projects/Spurious-Correlation/)** · *preprint, under submission* — with P. Kenfack
Most theory treats spurious correlations as **label-mediated** (the spurious feature depends only on the label). We analyse the harder **feature-mediated** case, where spurious and causal signals are entangled per-instance through group-specific operators.
We generalise the implicit-bias theorem of Soudry et al. (2018) to continuous distributions, then derive **closed-form group-wise error rates**: each group's error decays at a rate set by its hard margin. In the isotropic regime a single exponent governs the competition between the minority's geometric margin advantage and the coupling induced by spurious alignment — with a **phase transition**. This formalises in closed form the observation that group imbalance biases optimisation toward the majority, while showing the effect can be *entirely superseded* by margin advantages. → [code](https://github.com/Aser97/Class-Imbalance-And-Gradient-Descent-Dynamics)

**[Visual-TableQA: Open-Domain Benchmark for Reasoning over Table Images](https://arxiv.org/pdf/2509.07966)** · *NeurIPS 2025 Workshop*
Visual reasoning over structured tabular data remains one of the hardest challenges for VLMs. Visual-TableQA introduces a large-scale multimodal dataset of **LaTeX-rendered tables** and **9,000 reasoning-intensive QA pairs**, produced for under \$100 through a *multi-model collaborative generation pipeline* combining cross-model inspiration and LLM-jury filtering.

**[Modality-Swap Distillation: Rendering Textual Reasoning into Visual Supervision](https://github.com/AI-4-Everyone/Visual-TableQA-v2)** · *preprint*
A **modality-swap** approach for distilling reasoning ability from text-only LLMs into vision-language models: textual reasoning is *rendered* and re-used as synthetic visual supervision, improving cross-modal understanding at low cost.

**[Multi-Objective Representation for Numbers in Clinical Narratives](https://doi.org/10.48550/arXiv.2405.18448)** · *CoRR 2024*
Proposes **CamemBERT-Bio + LESA**, an efficient alternative to large-scale LLMs for modelling numerical magnitudes in medical text, with an F1 improvement while using less data and fewer parameters.

**[Parametric Graph for Unimodal Ranking Bandit](https://hal.archives-ouvertes.fr/hal-03256621/)** · *ICML 2021*
A *parametric multi-armed bandit* algorithm for ranking under structured feedback graphs, exploring the link between **reinforcement learning** and **online ranking** theory.

**Reviewer:** ICCV 2023 · CVPR 2026 · ICLR 2026

---

### 🧩 Projects

#### 🔍 [N2P — Making Learned Programs Explicit](https://github.com/galaxy532/N2P-Experiments)
Interpretability experiment suite on GPT-J / Llama-3-8B: helix and Fourier probes of number representations, activation and path patching, Edge-Pruning circuit discovery with Tracr ground truth. One run = one immutable results directory, one line in the run log.

#### 🛡️ [Youth Mental Health Safety Guardrail](https://github.com/Aser97/Guardrail)
Fine-tuned **Qwen2.5-7B + LoRA** input guardrail for youth crisis triage — 9-signal distress taxonomy, 2,532-row synthetic dataset built with CAMEL/PAIR adversarial generation, LR calibration head (F1=0.85 · Recall=0.88).

#### 🧾 [Visual-TableQA Pipeline](https://github.com/AI-4-Everyone/Visual-TableQA-v2)
Synthetic dataset generator for multimodal reasoning — creates 2,500 LaTeX tables + 9,000 QA pairs with LLM-jury quality control.

#### 🧬 [Multi-Objective Token Representation](https://github.com/sadc-lab/multiobjective_token_representation)
Implements the **LESA + Xval** framework for numerical reasoning in CamemBERT-Bio.

#### ⚖️ [Optimal Transport for Color Transfer](https://github.com/Aser97/Optimal-Transport)
Unbalanced OT visualizations for transferring color distributions between images.

#### ♟️ [Reinforcement Learning for Chess](https://github.com/Aser97/Chess)
C++ implementation of **SARSA** and **Monte Carlo** algorithms that learn to play chess through self-play and Stockfish simulations.
*(Reached ~1350 Elo after 1 hour of training.)*

---

### 🧠 Tech Stack

**Languages:** Python · C++ · JavaScript · Bash · HTML/CSS
**Frameworks:** PyTorch · Transformers · JAX · Hugging Face · vLLM · LoRA/PEFT
**Interpretability:** TransformerLens · SAELens · Edge Pruning · Tracr · activation & path patching
**Tools:** Docker · Weights & Biases · Paperspace · Git · MCP
**Focus Areas:** Mechanistic Interpretability · AI Safety · Learning Theory · Multimodal Reasoning

---

### 🌱 Beyond Research

When I'm not building models, I:
- Mentor students in **mathematics** and **AI fundamentals**
- Compete in **chess tournaments** ♟️ *(2000 Elo — [45th Chess Olympiad](https://aser97.github.io/Blog/jekyll/update/2024/10/02/chess-olympiad/))*
- Capture and edit **cinematic travel videos** 🎥
