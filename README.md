# Utilizing Local Hierarchy with Adversarial Training for Hierarchical Text Classification
This repository implements a adversarial framework for hierarchical text classification. This work has been accepted as the long paper "Utilizing Local Hierarchy with Adversarial Training for Hierarchical Text Classification" in LREC-COLING 2024.

## Requirements

* Python >= 3.6
* torch >= 1.8.0
* transformers >= 4.18.0
* datasets
* torch-geometric


## Preprocess

Please download the original dataset and then use these scripts.

### Web Of Science

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocessing code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocessing code here.
Please save the Excel data file `Data.xlsx` in `WebOfScience/Meta-data` as `Data.txt`.

```shell
cd data/WebOfScience
python preprocess_wos.py
python data_wos.py
```

### NYT

The original dataset can be acquired [here](https://catalog.ldc.upenn.edu/LDC2008T19).
Place the **unzipped** folder `nyt_corpus` inside `data/nyt` (or unzip `nyt_corpus_LDC2008T19.tgz` inside `data/nyt`).

```shell
cd data/nyt
# unzip if necessary
# tar -zxvf nyt_corpus_LDC2008T19.tgz -C ./
python data_nyt.py
```

### RCV1-V2

The preprocessing code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.
Place `rcv1.tar.xz` and `lyrl2004_tokens_train.dat` (can be downloaded [here](https://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz)) inside `data/rcv1`.

```shell
cd data/rcv1
python preprocess_rcv1.py ./
python data_rcv1.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] [--name NAME] [--update UPDATE] [--model MODEL] [--wandb] [--arch ARCH] [--seed SEED] [--loss LOSS] [--adv]

optional arguments:
  -h, --help                show this help message and exit
  --lr LR					Learning rate. Default: 3e-5.
  --data {WebOfScience,nyt,rcv1} Dataset.
  --batch BATCH             Batch size.
  --early-stop EARLY_STOP   Epoch before early stop.
  --device DEVICE           cuda or cpu. Default: cuda.
  --name NAME               A name for different runs.
  --update UPDATE           Gradient accumulate steps.
  --wandb                   Use wandb for logging.
  --seed SEED               Random seed.
  --model MODEL				"single_prompt" for HPT or "bert-new-htc" for HiBERT
  --adv						Use adversarial training framework.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

**Example:**
```shell
python train.py --name test --batch 8 --data nyt --model single_prompt --adv
```

### Reproducibility

We experiment on one GeForce RTX 3090 GPU (24G) with CUDA version $11.2$. We use a batch size of $8$ to fully tap one GPU.

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name nyt-test --batch 64
```
