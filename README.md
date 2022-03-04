multimodal machine translation(MMT) 
# Our dependency

* PyTorch version == 1.9.1
* Python version == 3.6.7
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1

# Install fairseq

```bash
cd fairseq_mmt
pip install --editable ./
```

# multi30k data & flickr30k entities
Multi30k data from [here](https://github.com/multi30k/dataset) and [here](https://www.statmt.org/wmt17/multimodal-task.html)  
flickr30k entities data from [here](https://github.com/BryanPlummer/flickr30k_entities)  
We get multi30k text data from [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)
```bash
# create a directory
flickr30k
├─ flickr30k-images
├─ test2017-images
├─ test_2016_flickr.txt
├─ test_2017_flickr.txt
├─ test_2017_mscoco.txt
├─ test_2018_flickr.txt
├─ testcoco-images
├─ train.txt
└─ val.txt
```

# Image feature
```bash
# please read scripts/README.md
python3 scripts/get_img_feat.py --dataset train
```

# Train and Test
```bash
sh preprocess.sh
sh train_mmt.sh
sh translation_mmt.sh
```

# Create masking data
```bash
pip3 install stanfordcorenlp 
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
cd fairseq_mmt
python3 record_masking_position.py 

cd data/masking
# create en-de masking data
python3 match_origin2bpe_position.py
python3 create_maskding1234_multi30k.py         # create mask1-4 data
python3 create_maskingcp_multi30k.py  # create mask color&people data

sh preprocess_mmt.sh
```
