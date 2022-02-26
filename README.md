# MoCoSE
Implementation of paper: [Exploring the Impact of Negative Samples of Contrastive Learning: A Case Study of Sentence Embedding]()
, been accepted to appear in the Findings of ACL 2022.

We propose a momentum contrastive learning model to sentence embedding, namely MoCoSE. We focus on the effect of negative queue length in text comparison learning.

Attention! You may need to:
1. download the [BERT weights](https://drive.google.com/file/d/1eG0zFgVH2PBBXUnYMgs_K9ODSV5ppoyd/view?usp=sharing) and change the path of the weights in demo. 
2. download the [sentEval](https://drive.google.com/file/d/1VNWVejfoLoZvmZOrmqnHw6Cbmffd4sWd/view?usp=sharing) and change the corresponding path in mocose_tools.

![architecture](architecture.png "Architecture of MoCoSE")

- mocose.py contains the main constituent code of the model;
- mocose_tools.py contains the code of the tools to evaluate the model;
- mocose_demo.ipynbs is the example code we provide for train and evaluation.

It is worth noting that our code is built on transformers 4.4.2 and some code may no longer be adapted in the last few versions of updates. If you want to reproduce the code, please install transformers 4.4.2 version. Second, you will need to modify the location of the SentEval in the tools code to be used for evaluating the dataset.

You can download MoCoSE-bert-base-uncased weights [HERE](https://drive.google.com/file/d/19eevBsaz8ApjgPfyx_hUtUsNlFYQ7riL/view?usp=sharing) .

STS Results in our paper:
| STS12      | STS13 | STS14 | STS15 | STS16 | STS-Benchmark | SICK-R | Avg. |
| ----------- | ----------- |----------- |----------- |----------- |----------- |----------- |----------- |
| 71.48      | 81.40       |74.47       |83.45       |78.99       |78.68       |72.44       |77.27       |
