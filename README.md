This repository contains source code for our NAACL 2021 paper "Improving the Lexical Ability of Pretrained Language Models for Unsupervised Neural
Machine Translation"  [(Arxiv preprint link)](https://arxiv.org/abs/2103.10531)

# Introduction 
Successful methods for unsupervised neural machine translation (UNMT)
employ cross-lingual pretraining via self-supervision (e.g. XLM, MASS, RE-LM), which requires
the model to align the lexical- and high-level representations of the two
languages. This is not effective for low-resource, distant languages.

Our method enhances the bilingual masked language model pretraining (of XLM, RE-LM) with lexical-level information 
by using type-level cross-lingual subword embeddings. 

Our method (**lexically-aligned** XLM/RE-LM) improves BLEU scores in UNMT by up to 4.5 points.
Bilingual lexicon induction results also show that our method works better compared to established UNMT baselines.
using our method compared to an established UNMT baseline.

This source code is largely based on [XLM](https://github.com/facebookresearch/XLM) and [RE-LM](https://github.com/alexandra-chron/relm_unmt).

# Prerequisites 

#### Dependencies

- Python 3.6.9
- [NumPy](http://www.numpy.org/) (tested on version 1.15.4)
- [PyTorch](http://pytorch.org/) (tested on version 1.2.0)
- [Apex](https://github.com/NVIDIA/apex#quick-start) (for fp16 training)

#### Install Requirements 
**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n relm python=3.6.9
conda activate relm
```

Install PyTorch ```1.2.0``` with the desired cuda version to use the GPU:

``` conda install pytorch==1.2.0 torchvision -c pytorch```

Clone the project:

```
git clone https://github.com/alexandra-chron/relm_unmt.git

cd relm_unmt
```


Then install the rest of the requirements:

```
pip install -r ./requirements.txt
```

It is **HIGHLY recommended** to use half precision (using [Apex](https://github.com/NVIDIA/apex#quick-start)) by simply adding `--fp16 True --amp 1` flags to each running command. Without it, you might run out of memory.

To train with multiple GPUs use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

To train with multiple GPUs and half precision use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --fp16 True --amp 1 
```

#### Download data 
We sample 68M English sentences from [Newscrawl](http://data.statmt.org/news-crawl/en/).

We use Macedonian and Albanian Common Crawl deduplicated monolingual data from the [OSCAR corpus](https://oscar-corpus.com/).

Our validation and test data is created by sampling from the  [SETIMES](http://opus.nlpl.eu/SETIMES.php) parallel En-Mk, En-Sq corpora. 
To allow reproducing our results, we provide the validation and test data in `./data/mk-en` and `./data/sq-en` directories.
## Preprocessing for pretraining

Before pretraining an HMR (high-monolingual-resource) monolingual MLM, make sure you
 have downloaded the HMR data and placed it in `./data/$HMR/` directory. 
 
 The data should be in the form:  `{train_raw, valid_raw, test_raw}.$HMR`. 

After that, run the following (example for En):
```
./get_data_mlm_pretraining.sh --src en
```


## RE-LM 

### 1. Train a monolingual LM 
Train your monolingual masked LM (BERT without the next-sentence prediction task) on the monolingual data:

```

python train.py                            \
--exp_name mono_mlm_en_68m                 \
--dump_path ./dumped                       \
--data_path ./data/en/                     \
--lgs 'en'                                 \
--mlm_steps 'en'                           \
--emb_dim 1024                             \
--n_layers 6                               \
--n_heads 8                                \
--dropout '0.1'                            \
--attention_dropout '0.1'                  \
--gelu_activation true                     \
--batch_size 32                            \
--bptt 256                                 \
--optimizer 'adam,lr=0.0001'               \
--epoch_size 200000                        \
--validation_metrics valid_en_mlm_ppl      \
--stopping_criterion 'valid_en_mlm_ppl,10' 

## There are other parameters that are not specified here (see train.py).
```

## Preprocessing for fine-tuning (and UNMT)

Before fine-tuning the pretrained MLM and running UNMT, make sure you
 have downloaded the LMR data and placed it in `./data/$LMR-$HMR/` directory. 
 
 The data should be in the form:  `{train_raw, valid_raw, test_raw}.$LMR`. 
 
 Then, run the following (example for En, Mk):
```
./get_data_and_preprocess.sh --src en --tgt mk
```

In Step 2, the embedding layer (and the output layer) of the MLM model will be increased by the amount of 
new items added to the existing vocabulary. 

In the directory `./data/$LMR-$HMR/`, a file named `vocab.$LMR-$HMR-ext-by-$NUMBER` has been created. 
This number indicates by how many items we need to extend the initial vocabulary, and consequently 
the embedding and linear layer, to account for the LMR language. 

You will need to give this value to the `--increase_vocab_by` argument so that you successfully run fine-tuning (step 2).  


### 2. Fine-tune it on both the LMR and HMR languages

```
python train.py                            \
--exp_name finetune_en_mlm_mk              \
--dump_path ./dumped/                      \
--reload_model 'mono_mlm_en_68m.pth'       \
--data_path ./data/mk-en/                  \
--lgs 'en-mk'                              \
--mlm_steps 'mk,en'                        \
--emb_dim 1024                             \
--n_layers 6                               \
--n_heads 8                                \
--dropout 0.1                              \
--attention_dropout 0.1                    \
--gelu_activation true                     \
--batch_size 32                            \
--bptt 256                                 \
--optimizer adam,lr=0.0001                 \
--epoch_size 50000                         \
--validation_metrics valid_mk_mlm_ppl      \
--stopping_criterion valid_mk_mlm_ppl,10   \
--increase_vocab_for_lang en               \
--increase_vocab_from_lang mk              \
--increase_vocab_by NUMBER #(see ./data/mk-en/vocab.mk-en-ext-by-$NUMBER)
```

### 3. Train a UNMT model (encoder and decoder initialized with RE-LM)

```
python train.py                            \
--exp_name unsupMT_ft_mk                   \
--dump_path ./dumped/                      \
--reload_model 'finetune_en_mlm_mk.pth,finetune_en_mlm_mk.pth' \
--data_path './data/mk-en'                 \
--lgs en-mk                                \
--ae_steps en,mk                           \
--bt_steps en-mk-en,mk-en-mk               \
--word_shuffle 3                           \
--word_dropout 0.1                         \
--word_blank 0.1                           \
--lambda_ae 0:1,100000:0.1,300000:0        \
--encoder_only False                       \
--emb_dim 1024                             \
--n_layers 6                               \
--n_heads 8                                \
--dropout 0.1                              \
--attention_dropout 0.1                    \
--gelu_activation true                     \
--tokens_per_batch 1000                    \
--batch_size 32                            \
--bptt 256                                 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 50000                         \
--eval_bleu true                           \
--stopping_criterion valid_mk-en_mt_bleu,10  \
--validation_metrics valid_mk-en_mt_bleu   \
--increase_vocab_for_lang en               \
--increase_vocab_from_lang mk

```


For the XLM baseline, follow the instructions in [XLM github page](https://github.com/facebookresearch/XLM)

If you use our work, please cite our paper: 

#### Reference

```
```
