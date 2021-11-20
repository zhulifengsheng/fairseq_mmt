multimodal machine translation(MMT) 
# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.


# Our dependency

* PyTorch version == 1.9.1
* Python version == 3.6.7
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1

# multi30k data & flickr30k entities
we use multi30k data from [here](https://github.com/LividWo/Revisit-MMT)  
The offical multi30k data from [here](https://github.com/multi30k/dataset)  
flickr30k entities data from [here](https://github.com/BryanPlummer/flickr30k_entities)

# VIT image feature
```bash
# please read scripts/README.md
python3 scripts/get_img_feat.py --dataset train
```

# preprocess text data
```bash
sh preprocess.sh
```

# train
```bash
sh train_mmt.sh
```

# translation
```bash
sh translation_mmt.sh
```
