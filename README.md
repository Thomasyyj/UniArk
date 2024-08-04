# UniArk
The repository for the NAACL 2024 paper "[UniArk: Improving Generalisation and Consistency for Factual Knowledge Extraction through Debiasing](https://arxiv.org/abs/2404.01253)".

Thanks for your attention. The code and dataset ParaTrex has been uploaded.

## Environments
To set up the required enviroments, use:

```
conda env create -f environment.yml 
```

## Usages
To run the experiments: 

```
bash train_bert.sh
```

## Data

The packed LAMA dataset is available at [[Zenodo](https://zenodo.org/record/5578210/files/P-tune_LAMA.tar.gz?download=1)].
s
If you use our packed up data, please download it and unzip it in the *data/* folder in the root directory.

## Acknowledge

We thank the implementation of [P-tuning](https://github.com/THUDM/P-tuning/tree/main/LAMA), which inspires some code in this repo.

## Citation
```
@article{yang2024uniark,
  title={UniArk: Improving Generalisation and Consistency for Factual Knowledge Extraction through Debiasing},
  author={Yang, Yijun and He, Jie and Chen, Pinzhen and Guti{\'e}rrez-Basulto, V{\'\i}ctor and Pan, Jeff Z},
  journal={arXiv preprint arXiv:2404.01253},
  year={2024}
}
```