This repository contains source code for our NAACL 2021 paper "Improving the Lexical Ability of Pretrained Language Models for Unsupervised Neural
Machine Translation"  [(Arxiv preprint link)](https://arxiv.org/abs/2103.10531)

# Model  
Successful methods for unsupervised neural machine translation (UNMT)
employ cross-lingual pretraining via self-supervision (e.g. XLM, MASS, RE-LM), which requires
the model to align the lexical- and high-level representations of the two
languages. This is not effective for low-resource, distant languages.

Our method enhances the bilingual masked language model pretraining (of XLM, RE-LM) with lexical-level information 
by using type-level cross-lingual subword embeddings. 

Our method (**lexically-aligned** XLM/RE-LM) improves BLEU scores in UNMT by up to 4.5 points.
Bilingual lexicon induction results also show that our method works better compared to established UNMT baselines.
using our method compared to an established UNMT baseline.

![lexical_fig-1](https://user-images.githubusercontent.com/30402550/113608477-eb0a5000-964a-11eb-8376-35ec98903025.jpg)

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
conda create -n lexical-lm python=3.6.9
conda activate lexical-lm
```

Install PyTorch ```1.2.0``` with the desired cuda version to use the GPU:

``` conda install pytorch==1.2.0 torchvision -c pytorch```

Clone the project:

```
git clone https://github.com/alexandra-chron/lexical_xlm_relm.git

cd lexical_xlm_relm
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

The validation and test data is provided in the [RE-LM](https://github.com/alexandra-chron/relm_unmt/tree/master/data) github repo.

# A) Lexical XLM 

### Preprocessing

Our preprocessing pipeline for XLM follows the one of the original [XLM](https://github.com/facebookresearch/XLM#1-preparing-the-data-1) repo. 

After the data is preprocessed, we train an XLM model, 
the embedding layer of which is initialized with aligned 
subword VecMap embeddings.

Before training the actual XLM, you need to learn fastText embeddings for the two corpora separately, after they have been split into subwords.

To do this, after cloning [fastText](https://github.com/facebookresearch/fastText) repo, run:

```
./fasttext skipgram -input ./data/mk-en/train.mk -output ./data/fasttext_1024.mk -dim 1024
./fasttext skipgram -input ./data/mk-en/train.en -output ./data/fasttext_1024.en -dim 1024
```
Now, you need to align the fastText vectors (without a seed dictionary, based on identical strings) using VecMap. After cloning its github repo ([VecMap](https://github.com/artetxem/vecmap)), run:

```
python3 ./vecmap/map_embeddings.py --identical fasttext_1024.en.vec fasttext_1024.mk.vec fasttext_1024.en.mapped.vec fasttext_1024.mk.mapped.vec --cuda 
```

Finally, simply concatenate the aligned vecmap vectors.

```cat fasttext_1024.en.mapped.vec fasttext_1024.mk.mapped.vec > fasttext_1024.en_mk.mapped.vec```

### 1. Train the lexically aligned XLM

```
python3 train.py 
--exp_name lexical_xlm_mk_en                  \
--dump_path './dumped'                        \
--data_path .data/mk-en-xlm                   \ 
--lgs 'mk-en'                                 \
--mlm_steps 'mk,en'                           \
--emb_dim 1024                                \
--n_layers 6                                  \
--n_heads 8                                   \
--dropout '0.1'                               \
--attention_dropout '0.1'                     \  
--gelu_activation true                        \
--batch_size 32                               \
--bptt 256                                    \
--optimizer 'adam,lr=0.0001'                  \ 
--epoch_size 200000                           \
--validation_metrics _valid_mlm_ppl           \ 
--stopping_criterion '_valid_mlm_ppl,10'      \ 
--reload_emb ./fasttext_1024.en_mk.mapped.vec 
```

### 2. Train a UNMT model (encoder and decoder initialized with lexically aligned XLM)

```
python train.py                                              \
--exp_name unsupMT_en-mk_lexical_xlm                         \
--dump_path ./dumped/                                        \
--reload_model 'lexical_xlm_mk_en.pth,lexical_xlm_mk_en.pth' \
--data_path './data/mk-en-xlm'                               \
--lgs en-mk                                                  \
--ae_steps en,mk                                             \
--bt_steps en-mk-en,mk-en-mk                                 \
--word_shuffle 3                                             \
--word_dropout 0.1                                           \
--word_blank 0.1                                             \
--lambda_ae 0:1,100000:0.1,300000:0                          \
--encoder_only False                                         \
--emb_dim 1024                                               \
--n_layers 6                                                 \
--n_heads 8                                                  \
--dropout 0.1                                                \
--attention_dropout 0.1                                      \
--gelu_activation true                                       \
--tokens_per_batch 1000                                      \
--batch_size 32                                              \
--bptt 256                                                   \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 50000                                           \
--eval_bleu true                                             \
--stopping_criterion valid_mk-en_mt_bleu,10                  \
--validation_metrics valid_mk-en_mt_bleu                     \

```

# B) Lexical RE-LM 

### Preprocessing 

Before pretraining an HMR (high-monolingual-resource) monolingual MLM, make sure you
 have downloaded the HMR data and placed it in `./data/$HMR/` directory. 
 
 The data should be in the form:  `{train_raw, valid_raw, test_raw}.$HMR`. 

After that, run the following (example for En):
```
./get_data_mlm_pretraining.sh --src en
```


### 1. Train a monolingual LM 

Pretrain a monolingual masked LM in a high-resource language as described in [RE-LM](https://github.com/alexandra-chron/relm_unmt#re-lm) and then preprocess the data used for fine-tuning following the same repo. 

### 2. Fine-tune it on both the LMR and HMR languages (add lexically-aligned embeddings)

```
python train.py                            \
--exp_name lexically_finetune_en_mlm_mk              \
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
--increase_vocab_by NUMBER #(see ./data/mk-en/vocab.mk-en-ext-by-$NUMBER) \
--reload_emb ./data/mk-en/fasttext_1024.en_mk_relm.mapped.vec              \
--relm_vecmap True
```

### 3. Train a UNMT model (encoder and decoder initialized with RE-LM)
Train a UNMT model as described in RE-LM. 

#Reference

If you use our work, please cite our paper: 

```
@misc{chronopoulou2021improving,
      title={Improving the Lexical Ability of Pretrained Language Models for Unsupervised Neural Machine Translation}, 
      author={Alexandra Chronopoulou and Dario Stojanovski and Alexander Fraser},
      year={2021},
      eprint={2103.10531},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
