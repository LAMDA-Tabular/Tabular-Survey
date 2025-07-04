# Representation Learning for Tabular Data: A Comprehensive Survey

Awesome Tabular Deep Learning for "[Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109)". If you use any content of this repo for your work, please cite the following bib entry: 

```
@article{jiang2025tabularsurvey,
         title={Representation Learning for Tabular Data: A Comprehensive Survey}, 
         author={Jun-Peng Jiang and
                 Si-Yang Liu and
                 Hao-Run Cai and
                 Qile Zhou and
                 Han-Jia Ye},
         journal={arXiv preprint arXiv:2504.16109},
         year={2025}
}
```

Feel free to [create new issues](https://github.com/LAMDA-Tabular/Tabular-Survey/issues/new) or [drop me an email](mailto:jiangjp@lamda.nju.edu.cn) if you find any interesting paper missing in our survey, and we shall include them in the next version.

## Updates

[04/2025] [arXiv](https://arxiv.org/abs/2504.16109) paper has been released.

[04/2025] The repository has been released.

## Introduction

Tabular data, structured as rows and columns, is among the most prevalent data types in machine learning classification and regression applications. Models for learning from tabular data have continuously evolved, with Deep Neural Networks (DNNs) recently demonstrating promising results through their capability of representation learning. 
In this survey, we systematically introduce the field of tabular representation learning, covering the background, challenges, and benchmarks, along with the pros and cons of using DNNs.
We organize existing methods into three main categories according to their generalization capabilities: **specialized, transferable, and general models**. **Specialized models** focus on tasks where training and evaluation occur within the same data distribution. We introduce a hierarchical taxonomy for specialized models based on the key aspects of tabular data—features, samples, and objectives—and delve into detailed strategies for obtaining high-quality feature- and sample-level representations.
**Transferable models** are pre-trained on one or more datasets and subsequently fine-tuned on downstream tasks, leveraging knowledge acquired from homogeneous or heterogeneous sources, or even cross-modalities such as vision and language. 
**General models**, also known as tabular foundation models, extend this concept further, allowing direct application to downstream tasks without additional fine-tuning. We group these general models based on the strategies used to adapt across heterogeneous datasets.
Additionally, we explore ensemble methods, which integrate the strengths of multiple tabular models. Finally, we discuss representative extensions of tabular learning, including open-environment tabular machine learning, multimodal learning with tabular data, and tabular understanding tasks.

<div align="center">
  <img src="resources/taxo.png" width="90%">
</div>
<div align="center">
  <img src="resources/Tabular_Deep_Learning.png" width="90%">
</div>

## Some Basic Resources
### Benchmarks

| Date | Name                                  | Paper                                                        | Publication | Code                                                         |
| ---- | ------------------------------------- | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| 2025 | TabArena                              | [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791) | CoRR        | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://tabarena.ai) |
| 2025 | MLE-Bench                             | [MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) | ICLR        | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/openai/mle-bench) |
| 2025 | TabReD                                | [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks](https://arxiv.org/abs/2406.19380) | ICLR        | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabred) |
| 2024 | Data-Centric Benchmark                | [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/atschalz/dc_tabeval) |
| 2024 | Better-by-Default                     | [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dholzmueller/pytabkit) |
| 2024 | LAMDA-Tabular-Bench                   | [A Closer Look at Deep Learning Methods on Tabular Datasets](https://arxiv.org/abs/2407.00956) | CoRR        | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/qile2000/LAMDA-TALENT) |
| 2024 | DMLR-ICLR24-Datasets-for-Benchmarking | [Towards Quantifying the Effect of Datasets for Benchmarking: A Look at Tabular Machine Learning](https://ml.informatik.uni-freiburg.de/wp-content/uploads/2024/04/61_towards_quantifying_the_effect.pdf) | DMLR        | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/automl/dmlr-iclr24-datasets-for-benchmarking) |
| 2023 | TableShift                            | [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/abs/2312.07577) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mlfoundations/tableshift) |
| 2023 | TabZilla                              | [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/abs/2305.02997) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/naszilla/tabzilla) |
|   2023  |EncoderBenchmarking |[A benchmark of categorical encoders for binary classification](https://arxiv.org/abs/2307.09191)|NeurIPS| [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/DrCohomology/EncoderBenchmarking)|
| 2022 | Grinsztajn et al. Benchmark           | [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/LeoGrin/tabular-benchmark) |
| 2021 | RTDL                                  | [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/rtdl-revisiting-models) |
| 2021 | WellTunedSimpleNets                   | [Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/abs/2106.11189) | NeurIPS     | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/machinelearningnuremberg/WellTunedSimpleNets) |

### Awesome Deep Tabular Toolboxs

- [**RTDL**](https://github.com/yandex-research/rtdl): A collection of papers and packages on deep learning for tabular data.
- [**TALENT**](https://github.com/qile2000/LAMDA-TALENT): A comprehensive toolkit and benchmark for tabular data learning, featuring 30 deep methods, more than 10 classical methods, and 300 diverse tabular datasets.
- [**pytorch_tabular**](https://github.com/manujosephv/pytorch_tabular): A standard framework for modelling Deep Learning Models for tabular data.
- [**pytorch-frame**](https://github.com/pyg-team/pytorch-frame): A modular deep learning framework for building neural network models on heterogeneous tabular data.
- [**DeepTables**](https://github.com/DataCanvasIO/DeepTables): An easy-to-use toolkit that enables deep learning to unleash great power on tabular data.
- [**AutoGluon**](https://github.com/autogluon/autogluon): A toolbox which automates machine learning tasks and enables to easily achieve strong predictive performance.
- ...

### Other Awesome Repositories
> **TabPFN and its extensions**
- [**TabPFN v1**](https://github.com/PriorLabs/TabPFN/tree/tabpfn_v1): [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848).
- [**TabPFN v2**](https://github.com/PriorLabs/TabPFN): [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6).
- [**TabPFN extensions**](https://github.com/PriorLabs/tabpfn-extensions).
- [**TabPFN-Time-Series**](https://github.com/PriorLabs/tabpfn-time-series): [The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features](https://arxiv.org/abs/2501.02945).
- [**TabICL**](https://github.com/soda-inria/tabicl): [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564).
- [**TICL**](https://github.com/microsoft/ticl):
  - [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/abs/2312.08598); 
  - [TabFlex: Scaling Tabular Learning to Millions with Linear Attention](https://openreview.net/forum?id=f8aganC0tN); 
  - [GAMFormer](https://arxiv.org/abs/2410.04560).
- ...

> **Some summary repositories**

- [Awesome-Tabular-LLMs](https://github.com/SpursGoZmy/Awesome-Tabular-LLMs)

- [Awesome-LLM-Tabular](https://github.com/johnnyhwu/Awesome-LLM-Tabular)

- [Tabular-LLM](https://github.com/SpursGoZmy/Tabular-LLM)

- [LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation)

- [Resources for Data Centric AI](https://github.com/HazyResearch/data-centric-ai)

- [TabSurvey](https://github.com/kathrinse/TabSurvey)



## Specialized Methods

| Date |      Name      |                            Paper                             |     Publication     |                             Code                             |
| -- | ------------ | ---------------------------------------------------------- | ----------------- | ---------------------------------------------------------- |
| 2025 |   ModernNCA    | [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](https://openreview.net/forum?id=JytL2MrlLT) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/qile2000/LAMDA-TALENT) |
| 2025 |   TabM    | [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabm) |
| 2024 |  ExcelFormer   | [Can a deep learning model be a sure bet for tabular prediction?](https://dl.acm.org/doi/abs/10.1145/3637528.3671893) |         KDD         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/whatashot/excelformer) |
| 2024 |    AMFormer    | [Arithmetic feature interaction is necessary for deep tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/29033) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/aigc-apps/AMFormer) |
| 2024 |     GRANDE     | [GRANDE: gradient-based decision tree ensembles for tabular data](https://openreview.net/forum?id=XEFWBxi075) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/s-marton/GRANDE) |
| 2024 |    DOFEN     | [DOFEN: Deep Oblivious Forest ENsemble](https://arxiv.org/abs/2412.16534) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Sinopac-Digital-Technology-Division/DOFEN) |
| 2024 |    RealMLP     | [Better by default: Strong pre-tuned mlps and boosted trees on tabular data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dholzmueller/pytabkit) |
| 2024 |     BiSHop     | [Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model](https://proceedings.mlr.press/v235/xu24l.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MAGICS-LAB/BiSHop) |
| 2024 |   SwitchTab    | [Switchtab: Switched autoencoders are effective tabular learners](https://ojs.aaai.org/index.php/AAAI/article/view/29523) |        AAAI         |                                                              |
| 2024 |     PTaRL      | [Ptarl: Prototype-based tabular representation learning via space calibration](https://openreview.net/forum?id=G32oY4Vnm8) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/HangtingYe/PTaRL) |
| 2024 |      TabR      | [Tabr: Tabular deep learning meets nearest neighbors in 2023](https://openreview.net/forum?id=rhgIgTSSxW) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabular-dl-tabr) |
| 2023 |                | [An inductive bias for tabular deep learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html) |       NeurIPS       |                                                              |
| 2023 |     TabRet     | [Tabret: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/abs/2303.15747) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/pfnet-research/tabret) |
| 2023 |     Trompt     | [Trompt: Towards a better deep neural network for tabular data](https://proceedings.mlr.press/v202/chen23c.html) |        ICML         |                                                              |
| 2023 |     TANGOS     | [Tangos: Regularizing tabular neural networks through gradient orthogonalization and specialization](https://openreview.net/forum?id=n6H86gW8u0d) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/alanjeffares/TANGOS) |
| 2022 |    MLP-PLR     | [On embeddings for numerical features in tabular deep learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Yura52/tabular-dl-num-embeddings) |
| 2022 |     SAINT      | [SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training](https://openreview.net/forum?id=FiyUTAy4sB8) |     NeurIPS WS      | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/somepago/saint) |
| 2022 |     DANets     | [Danets: Deep abstract networks for tabular data classification and regression](https://ojs.aaai.org/index.php/AAAI/article/view/20309) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/WhatAShot/DANet) |
| 2022 |      DNNR      | [DNNR: differential nearest neighbors regression](https://proceedings.mlr.press/v162/nader22a.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/younader/DNNR) |
| 2022 |    Hopular     | [Hopular: Modern hopfield networks for tabular data](https://arxiv.org/abs/2206.00664) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ml-jku/hopular) |
| 2022 |    LSPIN     | [Locally Sparse Neural Networks for Tabular Biomedical Data](https://proceedings.mlr.press/v162/yang22i.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jcyang34/lspin) |
| 2021 | Net-DNF | [Net-DNF: Effective Deep Modeling of Tabular Data](https://openreview.net/pdf?id=73WTGs96kho) |       ICLR      |  |
| 2021 | FT-Transformer | [Revisiting deep learning models for tabular data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/rtdl) |
| 2021 |     TabNet     | [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dreamquark-ai/tabnet) |
| 2021 |     DCNv2      | [DCN V2: improved deep & cross network and practical lessons for web-scale learning to rank systems](https://dl.acm.org/doi/abs/10.1145/3442381.3450078) |         WWW         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/CharlesShang/DCNv2) |
| 2021 |                | [Well-tuned simple nets excel on tabular datasets](https://proceedings.neurips.cc/paper/2021/hash/c902b497eb972281fb5b4e206db38ee6-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/machinelearningnuremberg/WellTunedSimpleNets) |
| 2021 |      NPT       | [Self-attention between datapoints: Going beyond individual input-output pairs in deep learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f1507aba9fc82ffa7cc7373c58f8a613-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/OATML/Non-Parametric-Transformers) |
| 2020 |                | [Survey on categorical data for neural networks](https://link.springer.com/article/10.1186/s40537-020-00305-w#auth-Taghi_M_-Khoshgoftaar-Aff1) | Journal of big data |                                                              |
| 2020 | TabTransformer | [Tabtransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/lucidrains/tab-transformer-pytorch) |
| 2020 |    GrowNet     | [Gradient boosting neural networks: Grownet](https://arxiv.org/abs/2002.07971) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sbadirli/GrowNet) |
| 2020 |      NODE      | [Neural oblivious decision ensembles for deep learning on tabular data](https://openreview.net/forum?id=r1eiu2VtwH) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Qwicen/node) |
| 2020 |      STG       | [Feature Selection using Stochastic Gates](https://proceedings.mlr.press/v119/yamada20a.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/runopti/stg) |
| 2019 |    AutoInt     | [Autoint: Automatic feature interaction learning via self-attentive neural networks](https://dl.acm.org/doi/abs/10.1145/3357384.3357925) |        CIKM         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/DeepGraphLearning/RecommenderSystems) |
| 2018 |      RLNs      | [Regularization learning networks: deep learning for tabular datasets](https://proceedings.neurips.cc/paper_files/paper/2018/hash/500e75a036dc2d7d2fec5da1b71d36cc-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/irashavitt/regularization_learning_networks) |
| 2017 |      SNN       | [Selfnormalizing neural networks](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html) |        NIPS         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/bioinf-jku/SNNs) |



## Transferable Methods

| Date | Name             | Paper                                                        | Publication        | Code                                                         |
| ---: | :--------------- | :----------------------------------------------------------- | :----------------- | :----------------------------------------------------------- |
| 2025 |                  | [A survey on self-supervised learning for non-sequential tabular data](https://link.springer.com/article/10.1007/s10994-024-06674-0) | Machine Learning   | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/wwweiwei/awesome-self-supervised-learning-for-tabular-data) |
| 2025 | Tab2Visual       | [Tab2Visual: Overcoming Limited Data in Tabular Data Classification Using Deep Learning with Visual Representations](https://arxiv.org/abs/2502.07181) | CoRR               |                                                              |
| 2024 | LFR              | [Self-supervised representation learning from random data projectors](https://openreview.net/forum?id=EpYnZpDpsQ) | ICLR               |                                                              |
| 2024 | UniTabE          | [UniTabE: A Universal Pretraining Protocol for Tabular Foundation Model in Data Science](https://openreview.net/forum?id=6LLho5X6xV) | ICLR               |                                                              |
| 2024 | CM2              | [Towards cross-table masked pretraining for web data mining](https://dl.acm.org/doi/abs/10.1145/3589334.3645707) | WWW                | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Chao-Ye/CM2) |
| 2024 | TP-BERTa         | [Making pre-trained language models great on tabular prediction](https://openreview.net/forum?id=anzIzGZuLi) | ICLR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jyansir/tp-berta) |
| 2024 | CARTE            | [CARTE: pretraining and transfer for tabular learning](https://proceedings.mlr.press/v235/kim24d.html) | ICML               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/soda-inria/carte) |
| 2024 | FeatLLM          | [Large language models can automatically engineer features for few-shot tabular learning](https://proceedings.mlr.press/v235/han24f.html) | ICML               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Sungwon-Han/FeatLLM) |
| 2024 | LM-IGTD          | [LM-IGTD: a 2d image generator for low-dimensional and mixed-type tabular data to leverage the potential of convolutional neural networks](https://arxiv.org/abs/2406.14566) | CoRR               |                                                              |
| 2023 | DoRA             | [Dora: Domain-based self-supervised learning framework for low-resource real estate appraisal](https://dl.acm.org/doi/abs/10.1145/3583780.3615470) | CIKM               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/wwweiwei/DoRA) |
| 2023 |                  | [Transfer learning with deep tabular models](https://openreview.net/forum?id=b0RuGUYo8pA) | ICLR               |                                                              |
| 2023 | ReConTab         | [Recontab: Regularized contrastive representation learning for tabular data](https://arxiv.org/abs/2310.18541) | CoRR               |                                                              |
| 2023 | TabRet           | [Tabret: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/abs/2303.15747) | CoRR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/pfnet-research/tabret) |
| 2023 | ORCA             | [Cross-modal fine-tuning: Align then refine](https://proceedings.mlr.press/v202/shen23e.html) | ICML               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sjunhongshen/ORCA) |
| 2023 | TabToken         | [Unlocking the transferability of tokens in deep models for tabular data](https://arxiv.org/abs/2310.15149) | CoRR               |                                                              |
| 2023 |                  | [Transfer learning with deep tabular models](https://openreview.net/forum?id=b0RuGUYo8pA) | ICLR               |                                                              |
| 2023 | Xtab             | [Xtab: Cross-table pretraining for tabular transformers](https://proceedings.mlr.press/v202/zhu23k.html) | ICML               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/BingzhaoZhu/XTab) |
| 2023 | Meta-Transformer | [Meta-transformer: A unified framework for multimodal learning](https://arxiv.org/abs/2307.10802) | CoRR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://kxgong.github.io/meta_transformer/) |
| 2023 | Binder           | [Binding language models in symbolic languages](https://openreview.net/forum?id=lH1PV42cbF) | ICLR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/xlang-ai/Binder) |
| 2023 | CAAFE            | [Large language models for automated data science: Introducing caafe for context-aware automated feature engineering](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html) | NeurIPS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/noahho/CAAFE) |
| 2023 | TaPTaP           | [Generative table pre-training empowers models for tabular prediction](https://openreview.net/forum?id=3gdG9upo7e) | EMNLP              | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ZhangTP1996/TapTap) |
| 2023 | TabLLM           | [Tabllm: few-shot classification of tabular data with large language models](https://proceedings.mlr.press/v206/hegselmann23a.html) | AISTATS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/clinicalml/TabLLM) |
| 2023 | UniPredict       | [Unipredict: Large language models are universal tabular predictors](https://arxiv.org/abs/2310.03266) | CoRR               |                                                              |
| 2023 | TablEye          | [Tableye: Seeing small tables through the lens of images](https://arxiv.org/abs/2307.02491) | CoRR               |                                                              |
| 2022 |                  | [Revisiting pretraining objectives for tabular deep learning](https://arxiv.org/abs/2207.03208) | CoRR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/puhsu/tabular-dl-pretrain-objectives) |
| 2022 | SEFS             | [Self-supervision enhanced feature selection with correlated gates](https://openreview.net/forum?id=oDFvtxzPOx) | ICLR               | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/chl8856/SEFS) |
| 2022 | MET              | [MET: masked encoding for tabular data](https://arxiv.org/abs/2206.08564) | CoRR               |                                                              |
| 2022 | SAINT            | [SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training](https://openreview.net/forum?id=FiyUTAy4sB8) | NeurIPS WS         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/somepago/saint) |
| 2022 | SCARF            | [Scarf: Self-supervised contrastive learning using random feature corruption](https://openreview.net/forum?id=CuV_qYkmKb3) | ICLR               |                                                              |
| 2022 | Stab             | [Stab: Self-supervised learning for tabular data](https://openreview.net/forum?id=EfR55bFcrcI) | NeurIPS WS         |                                                              |
| 2022 | DEN              | [Distribution embedding networks for generalization from a diverse set of classification tasks](https://openreview.net/forum?id=F2rG2CXsgO) |                    |                                                              |
| 2022 | TransTab         | [Transtab: Learning transferable tabular transformers across tables](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1377f76686d56439a2bd7a91859972f5-Abstract-Conference.html) | NeurIPS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/RyanWangZf/transtab) |
| 2022 | Ptab             | [Ptab: Using the pre-trained language model for modeling tabular data](https://arxiv.org/abs/2209.08060) | CoRR               |                                                              |
| 2022 | LIFT             | [LIFT: language-interfaced fine-tuning for non-language machine learning tasks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ce7fe1d2730f53cb3857032952cd1b8-Abstract-Conference.html) | NeurIPS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/UW-Madison-Lee-Lab/LanguageInterfacedFineTuning) |
| 2021 | SubTab           | [Subtab: Subsetting features of tabular data for self-supervised representation learning](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html) | NeurIPS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AstraZeneca/SubTab) |
| 2021 | DACL             | [Towards domain-agnostic contrastive learning](http://proceedings.mlr.press/v139/verma21a.html) | ICML               |                                                              |
| 2021 | IGTD             | [Converting tabular data into images for deep learning with convolutional neural networks](https://www.nature.com/articles/s41598-021-90923-y) | Scientific reports | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/zhuyitan/IGTD) |
| 2020 | VIME             | [VIME: extending the success of self- and semi-supervised learning to tabular domain](https://proceedings.neurips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) | NeurIPS            | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jsyoon0823/VIME) |
| 2020 |                  | [Meta-learning from tasks with heterogeneous attribute spaces](https://proceedings.neurips.cc/paper/2020/hash/438124b4c06f3a5caffab2c07863b617-Abstract.html) | NeurIPS            |                                                              |
| 2020 | TAC              | [A novel method for classification of tabular data using convolutional neural networks](https://www.biorxiv.org/content/10.1101/2020.05.02.074203.abstract) | Biorxiv            |                                                              |
| 2019 | Super-TML        | [Supertml: Two-dimensional word embedding for the precognition on structured tabular data](http://openaccess.thecvf.com/content_CVPRW_2019/html/Precognition/Sun_SuperTML_Two-Dimensional_Word_Embedding_for_the_Precognition_on_Structured_Tabular_CVPRW_2019_paper.html) | CVPR WS            |                                                              |



## General Methods

| Date | Name          | Paper                                                        | Publication | Code                                                         |
| ---: | :------------ | :----------------------------------------------------------- | :---------- | :----------------------------------------------------------- |
| 2025 | Beta*         | [Tabpfn unleashed: A scalable and effective solution to tabular classification problems](https://arxiv.org/abs/2502.02527) | ICML        |                                                              |
| 2025 | MotherNet     | [MotherNet: Fast Training and Inference via Hyper-Network Transformers](https://openreview.net/forum?id=6H4jRWKFc3&noteId=qln8G23j4b) | ICLR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/ticl?tab=readme-ov-file#MotherNet) |
| 2025 | TabPFN v2     | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | Nature      | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2025 | TabForestPFN* | [Fine-tuned in-context learning transformers are excellent tabular data classifiers](https://arxiv.org/abs/2405.13396) | CoRR        |                                                              |
| 2025 | APT*          | [Zero-shot meta-learning for tabular prediction tasks with adversarially pre-trained transformer](https://arxiv.org/abs/2502.04573) | CoRR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yulun-rayn/APT) |
| 2025 | TabICL*       | [Tabicl: A tabular foundation model for in-context learning on large data](https://arxiv.org/abs/2502.05564) | ICML        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/soda-inria/tabicl) |
| 2025 | EquiTabPFN*   | [Equitabpfn: A targetpermutation equivariant prior fitted networks](https://arxiv.org/abs/2502.06684) | CoRR        |                                                              |
| 2025 | *             | [Scalable in-context learning on tabular data via retrieval-augmented large language models](https://arxiv.org/abs/2502.03147) | CoRR        |                                                              |
| 2024 | HyperFast     | [Hyperfast: Instant classification for tabular data](https://ojs.aaai.org/index.php/AAAI/article/view/28988) | AAAI        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AI-sandbox/HyperFast) |
| 2024 | TabDPT*       | [Tabdpt: Scaling tabular foundation models](https://arxiv.org/pdf/2410.18164?) | CoRR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/TabDPT) |
| 2024 | MIXTUREPFN*   | [Mixture of incontext prompters for tabular pfns](https://arxiv.org/abs/2405.16156) | CoRR        |                                                              |
| 2024 | LoCalPFN*     | [Retrieval & fine-tuning for in-context tabular models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html) | NeurIPS     | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/LoCalPFN) |
| 2024 | LE-TabPFN*    | [Towards localization via data embedding for tabPFN](https://openreview.net/forum?id=LFyQyV5HxQ) | NeurIPS WS  |                                                              |
| 2024 | TabFlex*      | [Tabflex: Scaling tabular learning to millions with linear attention](https://openreview.net/forum?id=f8aganC0tN) | NeurIPS WS  |                                                              |
| 2024 | *             | [Exploration of autoregressive models for in-context learning on tabular data](https://openreview.net/forum?id=4dOJ0PRY7R) | NeurIPS WS  |                                                              |
| 2024 | TabuLa-8B     | [Large scale transfer learning for tabular data via language modeling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) | NeurIPS     | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mlfoundations/rtfm) |
| 2024 | GTL           | [From supervised to generative: A novel paradigm for tabular deep learning with large language models](https://dl.acm.org/doi/abs/10.1145/3637528.3671975) | KDD         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/Industrial-Foundation-Models) |
| 2024 | MediTab       | [Meditab: Scaling medical tabular data predictors via data consolidation, enrichment, and refinement](https://www.ijcai.org/proceedings/2024/0670.pdf) | IJCAI       |                                                              |
| 2023 | TabPTM        | [Training-free generalization on heterogeneous tabular data via meta-representation](https://arxiv.org/abs/2311.00055) | CoRR        |                                                              |
| 2023 | TabPFN        | [Tabpfn: A transformer that solves small tabular classification problems in a second](https://openreview.net/forum?id=cp5PvcI6w8_) | ICLR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |

\* denotes that the method is a variation of TabPFN, some of which requires fine-tuning for downstream tasks.



## Ensemble Methods

| Date | Name          | Paper                                                        | Publication | Code                                                         |
| ---: | :------------ | :----------------------------------------------------------- | :---------- | :----------------------------------------------------------- |
| 2025 |   TabM    | [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabm) |
| 2025 | TabPFN v2 | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | Nature      | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2025 |   Beta    | [Tabpfn unleashed: A scalable and effective solution to tabular classification problems](https://arxiv.org/abs/2502.02527) | CoRR        |    |
| 2025 | LLM-Boost, PFN-Boost | [Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes](https://arxiv.org/abs/2502.02672) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MayukaJ/LLM-Boost) |
| 2024 | HyperFast | [Hyperfast: Instant classification for tabular data](https://ojs.aaai.org/index.php/AAAI/article/view/28988) | AAAI        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AI-sandbox/HyperFast) |
| 2024 |  GRANDE   | [GRANDE: gradient-based decision tree ensembles for tabular data](https://openreview.net/forum?id=XEFWBxi075) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/s-marton/GRANDE) |
| 2023 |   TabPTM    | [Training-free generalization on heterogeneous tabular data via meta-representation](https://arxiv.org/abs/2311.00055) |        CoRR         |    |
| 2023 |   TabPFN    | [Tabpfn: A transformer that solves small tabular classification problems in a second](https://openreview.net/forum?id=cp5PvcI6w8_) | ICLR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2020 | TabTransformer | [Tabtransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/lucidrains/tab-transformer-pytorch) |
| 2020 |  GrowNet  | [Gradient boosting neural networks: Grownet](https://arxiv.org/abs/2002.07971) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sbadirli/GrowNet) |
| 2020 |   NODE    | [Neural oblivious decision ensembles for deep learning on tabular data](https://openreview.net/forum?id=r1eiu2VtwH) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Qwicen/node) |



## Extensions

> **Clustering**

- [Interpretable Deep Clustering for Tabular Data](https://proceedings.mlr.press/v235/svirsky24a.html)
- [Tabledc: Deep clustering for tabular data](https://arxiv.org/abs/2405.17723)
- ...

> **Anomaly Detection**

- [Anomaly detection for tabular data with internal contrastive learning](https://openreview.net/forum?id=_hszZbt46bT)
- [Adbench: Anomaly detection benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html)
- [Anomaly detection of tabular data using llms](https://arxiv.org/abs/2406.16308)
- [Anomaly Detection with Variance Stabilized Density Estimation](https://arxiv.org/abs/2306.00582)
- [Transductive and Inductive Outlier Detection with Robust Autoencoders](https://openreview.net/forum?id=UkA5dZs5mP)
- ...

> **Tabular Generation**

- [Codi: Co-evolving contrastive diffusion models for mixed-type tabular synthesis](http://proceedings.mlr.press/v202/lee23i.html)
- [Causality for tabular data synthesis: A high-order structure causal benchmark framework](https://arxiv.org/abs/2406.08311)
- [Generating new concepts with hybrid neuro-symbolic models](https://arxiv.org/abs/2003.08978)
- ...

> **Interpretability**

- [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826)
- [Tabtransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678)
- [Revisiting deep learning models for tabular data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)
- [Neural oblivious decision ensembles for deep learning on tabular data](https://openreview.net/forum?id=r1eiu2VtwH)
- ...

> **Open-Environment Tabular Machine Learning**

- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)	
- [Driftresilient tabpfn: In-context learning temporal distribution shifts on tabular data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b2e2774c8e76afe191b5bf518f5cb727-Abstract-Conference.html)
- [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/abs/2312.07577)
- [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks](https://arxiv.org/abs/2406.19380)
- [Understanding the limits of deep tabular methods with temporal shift](https://arxiv.org/abs/2502.20260)
- ...

> **Multi-modal Learning with Tabular Data**

- [Best of both worlds: Multimodal contrastive learning with tabular and imaging data](http://openaccess.thecvf.com/content/CVPR2023/html/Hager_Best_of_Both_Worlds_Multimodal_Contrastive_Learning_With_Tabular_and_CVPR_2023_paper.html)
- [Tabular insights, visual impacts: Transferring expertise from tables to images](https://proceedings.mlr.press/v235/jiang24h.html)
- [Tip: Tabular-image pre-training for multimodal classification with incomplete data](https://arxiv.org/abs/2407.07582)
- ...

> **Tabular Understanding**

- [Tablebank: Table benchmark for image-based table detection and recognition](https://arxiv.org/abs/1903.01949)
- [Multimodalqa: Complex question answering over text, tables and images](https://arxiv.org/abs/2104.06039)
- [Donut: Document understanding transformer without ocr](https://arxiv.org/abs/2111.15664)
- [Monkey: Image resolution and text label are important things for large multi-modal models](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Monkey_Image_Resolution_and_Text_Label_Are_Important_Things_for_CVPR_2024_paper.pdf)
- [mPLUG-DocOwl: Modularized multimodal large language model for document understanding](https://arxiv.org/pdf/2307.02499)
- [Tabpedia: Towards comprehensive visual table understanding with concept synergy](https://proceedings.neurips.cc/paper_files/paper/2024/file/0d97fe65d7a1dc12a05642d9fa4cd578-Paper-Conference.pdf)
- [Compositional Condition Question Answering in Tabular Understanding](https://openreview.net/pdf?id=aXU48nrA2v)
- [Multimodal Tabular Reasoning with Privileged Structured Information](https://arxiv.org/abs/2506.04088)
- ...

Please refer to [Awesome-Tabular-LLMs](https://github.com/SpursGoZmy/Awesome-Tabular-LLMs) for more information.



## Workshops

- [Table Representation Learning Workshop @ NeurIPS 2022](https://neurips.cc/virtual/2022/workshop/49995)
- [Table Representation Learning Workshop @ NeurIPS 2023](https://neurips.cc/virtual/2023/workshop/66546)
- [Table Representation Learning Workshop @ NeurIPS 2024](https://table-representation-learning.github.io/)
- [Table Representation Learning Workshop @ ACL 2025](https://table-representation-learning.github.io/ACL2025/)
- [DExLLM workshop @ ICDE'25](https://advanced-recommender-systems.github.io/Data-meets-LLMs/)
- [LLM + Vector Data @ ICDE 2025](https://llmvdb.github.io/)
- [Data-AI Systems (DAIS) @ ICDE 2025](https://dais-workshop-icde.github.io/)
- [Frontiers of DE & AI @ ICDE 2025](https://fdea-workshop.github.io/)
- [Foundation Models for Structured Data @ ICML 2025](https://icml-structured-fm-workshop.github.io/)



## Acknowledgment

This repo is modified from [TALENT](https://github.com/LAMDA-Tabular/TALENT).

## Correspondence

This repo is developed and maintained by [Jun-Peng Jiang](https://www.lamda.nju.edu.cn/jiangjp/), [Si-Yang Liu](https://www.lamda.nju.edu.cn/liusy/), Hao-Run Cai, [Qile Zhou](https://www.lamda.nju.edu.cn/zhouql/), and [Han-Jia Ye](https://www.lamda.nju.edu.cn/yehj/). If you have any questions, please feel free to contact us by opening new issues or email:

- Jun-Peng Jiang: jiangjp@lamda.nju.edu.cn
- Si-Yang Liu: liusy@lamda.nju.edu.cn
- Hao-Run Cai: caihr@smail.nju.edu.cn
- Qile Zhou: zhouql@lamda.nju.edu.cn
- Han-Jia Ye: yehj@lamda.nju.edu.cn
