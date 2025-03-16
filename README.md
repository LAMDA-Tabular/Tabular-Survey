# Representation Learning for Tabular Data: A Comprehensive Survey

Awesome Tabular Deep Learning for "[Representation Learning for Tabular Data: A Comprehensive Survey](http://arxiv.org/abs/2302.03648)". If you use any content of this repo for your work, please cite the following bib entry: 

    @article{xxx
     }

Feel free to [create new issues](xxx) or [drop me an email](mailto:jiangjp@lamda.nju.edu.cn) if you find any interesting paper missing in our survey, and we shall include them in the next version.

## Updates

[02/2023] [arXiv](xxxx) paper has been released.

[02/2023] The repository has been released.

## Other Resources

- [TALENT](https://github.com/qile2000/LAMDA-TALENT): A comprehensive machine learning toolbox
- [Benchmark](https://arxiv.org/abs/2407.00956): A Closer Look at Deep Learning Methods on Tabular Datasets; Datasets are available at [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link)
- [Baseline](https://arxiv.org/abs/2407.03257): Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later

## Introduction

Tabular data remains a cornerstone of machine learning applications across diverse fields, yet its structured format presents unique challenges and opportunities for model development. This survey systematically examines the evolutionary trajectory of tabular data learning methods, spanning from conventional machine learning paradigms to contemporary deep learning architectures. We classify existing deep tabular methods into three categories according to their increasing generalization capabilities: specialized methods, transferable methods, and general methods. We define the taxonomy of fundamental specialized methods through the components of row-column-label in raw data, focusing on: 1) feature representation enhancement, 2) sample relationship modeling, and 3) objective optimization or regularization. Building upon this foundation, the survey investigates two transformative paradigms in tabular data: the transferable methods leverage the advantage of pre-trained models to adapt quickly to downstream tasks with fine-tuning, and the general models capable of zero-shot generalization across heterogeneous datasets, both of which take advantage of the representation ability of deep neural networks and make the model shareable across heterogeneous tabular datasets. Finally, we explore extensions of tabular learning to complex tasks, including clustering, anomaly detection, tabular generation, interpretability, open-environment tabular machine learning, multimodal learning with tabular data, and tabular understanding. We conclude by addressing open problems and directions for future research in the field, aiming to guide advancements in learning with tabular data.

<div align="center">
  <img src="resources/taxonomy_fig.png" width="90%">
<img src="resources/roadmap.png" width="90%">
</div>


## Specialized Methods

| Date |      Name      |                            Paper                             |     Publication     |                             Code                             |
| :--: | :------------: | :----------------------------------------------------------: | :-----------------: | :----------------------------------------------------------: |
| 2025 |   ModernNCA    | [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](https://openreview.net/forum?id=JytL2MrlLT) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/qile2000/LAMDA-TALENT) |
| 2024 |  ExcelFormer   | [Can a deep learning model be a sure bet for tabular prediction?](https://dl.acm.org/doi/abs/10.1145/3637528.3671893) |         KDD         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/whatashot/excelformer) |
| 2024 |    AMFormer    | [Arithmetic feature interaction is necessary for deep tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/29033) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/aigc-apps/AMFormer) |
| 2024 |     GRANDE     | [GRANDE: gradient-based decision tree ensembles for tabular data](https://openreview.net/forum?id=XEFWBxi075) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/s-marton/GRANDE) |
| 2024 |    RealMLP     | [Better by default: Strong pre-tuned mlps and boosted trees on tabular data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dholzmueller/pytabkit) |
| 2024 |     BiSHop     | [Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model](https://proceedings.mlr.press/v235/xu24l.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MAGICS-LAB/BiSHop) |
| 2024 |   SwitchTab    | [Switchtab: Switched autoencoders are effective tabular learners](https://ojs.aaai.org/index.php/AAAI/article/view/29523) |        AAAI         |                                                              |
| 2024 |     PTaRL      | [Ptarl: Prototype-based tabular representation learning via space calibration](https://openreview.net/forum?id=G32oY4Vnm8) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/HangtingYe/PTaRL) |
| 2024 |      TabR      | [Tabr: Tabular deep learning meets nearest neighbors in 2023](https://openreview.net/forum?id=rhgIgTSSxW) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabular-dl-tabr) |
| 2023 |                | [An inductive bias for tabular deep learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html) |       NeurIPS       |                                                              |
| 2023 |     TabRet     | [Tabret: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/abs/2303.15747) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/pfnet-research/tabret) |
| 2023 |      Xtab      | [Xtab: Cross-table pretraining for tabular transformers](https://proceedings.mlr.press/v202/zhu23k.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/BingzhaoZhu/XTab) |
| 2023 |     Trompt     | [Trompt: Towards a better deep neural network for tabular data](https://proceedings.mlr.press/v202/chen23c.html) |        ICML         |                                                              |
| 2023 |     TANGOS     | [Tangos: Regularizing tabular neural networks through gradient orthogonalization and specialization](https://openreview.net/forum?id=n6H86gW8u0d) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/alanjeffares/TANGOS) |
| 2022 |    MLP-PLR     | [On embeddings for numerical features in tabular deep learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Yura52/tabular-dl-num-embeddings) |
| 2022 |     SAINT      | [SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training](https://openreview.net/forum?id=FiyUTAy4sB8) |     NeurIPS WS      | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/somepago/saint) |
| 2022 |     DANets     | [Danets: Deep abstract networks for tabular data classification and regression](https://ojs.aaai.org/index.php/AAAI/article/view/20309) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github. com/WhatAShot/DANet) |
| 2022 |      DNNR      | [DNNR: differential nearest neighbors regression](https://proceedings.mlr.press/v162/nader22a.html) |        ICML         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/younader/DNNR) |
| 2021 | FT-Transformer | [Revisiting deep learning models for tabular data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/rtdl) |
| 2021 |     TabNet     | [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826) |        AAAI         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dreamquark-ai/tabnet) |
| 2021 |     DCNv2      | [DCN V2: improved deep & cross network and practical lessons for web-scale learning to rank systems](https://dl.acm.org/doi/abs/10.1145/3442381.3450078) |         WWW         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/CharlesShang/DCNv2) |
| 2021 |                | [Well-tuned simple nets excel on tabular datasets](https://proceedings.neurips.cc/paper/2021/hash/c902b497eb972281fb5b4e206db38ee6-Abstract.html) |       NeurIPS       | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/machinelearningnuremberg/WellTunedSimpleNets) |
| 2020 |                | [Survey on categorical data for neural networks](https://link.springer.com/article/10.1186/s40537-020-00305-w#auth-Taghi_M_-Khoshgoftaar-Aff1) | Journal of big data |                                                              |
| 2020 | TabTransformer | [Tabtransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/lucidrains/tab-transformer-pytorch) |
| 2020 |    GrowNet     | [Gradient boosting neural networks: Grownet](https://arxiv.org/abs/2002.07971) |        CoRR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sbadirli/GrowNet) |
| 2020 |      NODE      | [Neural oblivious decision ensembles for deep learning on tabular data](https://openreview.net/forum?id=r1eiu2VtwH) |        ICLR         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Qwicen/node) |
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

| Date | Name         | Paper                                                        | Publication | Code                                                         |
| ---: | :----------- | :----------------------------------------------------------- | :---------- | :----------------------------------------------------------- |
| 2025 | MotherNet    | [MotherNet: Fast Training and Inference via Hyper-Network Transformers](https://openreview.net/forum?id=6H4jRWKFc3&noteId=qln8G23j4b) | ICLR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/ticl?tab=readme-ov-file#MotherNet) |
| 2025 | TabPFN v2    | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | Nature      | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2025 | TabForestPFN | [Fine-tuned in-context learning transformers are excellent tabular data classifiers](https://arxiv.org/abs/2405.13396) | CoRR        |                                                              |
| 2025 | APT          | [Zero-shot meta-learning for tabular prediction tasks with adversarially pre-trained transformer](https://arxiv.org/abs/2502.04573) | CoRR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yulun-rayn/APT) |
| 2025 | TabICL       | [Tabicl: A tabular foundation model for in-context learning on large data](https://arxiv.org/abs/2502.05564) | CoRR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/soda-inria/tabicl) |
| 2025 | Beta         | [Tabpfn unleashed: A scalable and effective solution to tabular classification problems](https://arxiv.org/abs/2502.02527) | CoRR        |                                                              |
| 2025 | EquiTabPFN   | [Equitabpfn: A targetpermutation equivariant prior fitted networks](https://arxiv.org/abs/2502.06684) | CoRR        |                                                              |
| 2025 |              | [Scalable in-context learning on tabular data via retrieval-augmented large language models](https://arxiv.org/abs/2502.03147) | CoRR        |                                                              |
| 2024 | HyperFast    | [Hyperfast: Instant classification for tabular data](https://ojs.aaai.org/index.php/AAAI/article/view/28988) | AAAI        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AI-sandbox/HyperFast) |
| 2024 | TabDPT       | [Tabdpt: Scaling tabular foundation models](https://arxiv.org/pdf/2410.18164?) | CoRR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/TabDPT) |
| 2024 | MIXTUREPFN   | [Mixture of incontext prompters for tabular pfns](https://arxiv.org/abs/2405.16156) | CoRR        |                                                              |
| 2024 | LoCalPFN     | [Retrieval & fine-tuning for in-context tabular models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html) | NeurIPS     | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/LoCalPFN) |
| 2024 | LE-TabPFN    | [Towards localization via data embedding for tabPFN](https://openreview.net/forum?id=LFyQyV5HxQ) | NeurIPS WS  |                                                              |
| 2024 | TabFlex      | [Tabflex: Scaling tabular learning to millions with linear attention](https://openreview.net/forum?id=f8aganC0tN) | NeurIPS WS  |                                                              |
| 2024 |              | [Exploration of autoregressive models for in-context learning on tabular data](https://openreview.net/forum?id=4dOJ0PRY7R) | NeurIPS WS  |                                                              |
| 2024 | TabuLa-8B    | [Large scale transfer learning for tabular data via language modeling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) | NeurIPS     | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mlfoundations/rtfm) |
| 2024 | GTL          | [From supervised to generative: A novel paradigm for tabular deep learning with large language models](https://dl.acm.org/doi/abs/10.1145/3637528.3671975) | KDD         | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/Industrial-Foundation-Models) |
| 2024 | MediTab      | [Meditab: Scaling medical tabular data predictors via data consolidation, enrichment, and refinement](https://www.ijcai.org/proceedings/2024/0670.pdf) | IJCAI       |                                                              |
| 2023 | TabPTM       | [Training-free generalization on heterogeneous tabular data via meta-representation](https://arxiv.org/abs/2311.00055) | CoRR        |                                                              |
| 2023 | TabPFN       | [Tabpfn: A transformer that solves small tabular classification problems in a second](https://openreview.net/forum?id=cp5PvcI6w8_) | ICLR        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |



## Workshops

- [Table Representation Learning Workshop @ NeurIPS 2023](https://neurips.cc/virtual/2023/workshop/66546)
- [Table Representation Learning Workshop @ NeurIPS 2024](https://table-representation-learning.github.io/)

## Acknowledgment

This repo is modified from [TALENT](https://github.com/qile2000/LAMDA-TALENT).

## Correspondence

This repo is developed and maintained by [Jun-Peng Jiang](https://www.lamda.nju.edu.cn/jiangjp/), [Siyang Liu](https://www.lamda.nju.edu.cn/liusy/), Hao-Run Cai, and [Han-Jia Ye](https://www.lamda.nju.edu.cn/yehj/). If you have any questions, please feel free to contact us by opening new issues or email:

- Jun-Peng Jiang: jiangjp@lamda.nju.edu.cn
- Siyang Liu: liusy@lamda.nju.edu.cn
- Hao-Run Cai: caihr@smail.nju.edu.cn
- Han-Jia Ye: yehj@lamda.nju.edu.cn
