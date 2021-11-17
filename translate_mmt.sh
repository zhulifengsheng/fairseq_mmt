#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
task=multi30k-en2de
#task=multi30k-en2fr

if [ $task == "multi30k-en2de" ]; then
	tgt_lang=de
elif [ $task == "multi30k-en2fr" ]; then
	tgt_lang=fr
fi
who=test
random_image_translation=0
length_penalty=0.8
#length_penalty=1.6
#length_penalty=2.2
#length_penalty=3.2

# set tag
#model_dir_tag=vit_small_patch16_224/128-outside-vit_small_patch16_224-mask4-imgdrop0.2

#model_dir_tag=vit_base_patch32_224_in21k/128-outside-vit_base_patch32_224_in21k-mask6-imgdrop0.0
#model_dir_tag=queryInst/seed2-outside-queryInst-mask0-imgdrop0.2
#model_dir_tag=detr_resnet101_dc5/seed1-1outside-detr_resnet101_dc5-mask0-imgdrop0.2
model_dir_tag=catr_finetune_decoder_out/seed4-1outside-catr_finetune_decoder_out-mask0-imgdrop0.2
#model_dir_tag=CNN/128-outside-CNN-mask6-imgdrop0.0
#model_dir_tag=swin_base/128-outside-swin_base-mask6-imgdrop0.0/
#model_dir_tag=swin_small/128-outside-swin_small-mask6-imgdrop0.0/
#model_dir_tag=swin_tiny/128-outside-swin_tiny-mask6-imgdrop0.0/

# get tag
array=(${model_dir_tag//-/ })
n1=$(( ${#array[@]} - 2 ))
which_mask=${array[n1]}
which_data=${array[n1-1]}

if [ $which_mask == "mask0" ]; then
        data_dir=multi30k.en-de
        #data_dir=multi30k.en-fr
elif [ $which_mask == "mask1" ]; then
        data_dir=multi30k.en-de.mask1
        #data_dir=multi30k.en-fr.mask1
elif [ $which_mask == "mask2" ]; then
        data_dir=multi30k.en-de.mask2
        #data_dir=multi30k.en-fr.mask2
elif [ $which_mask == "mask3" ]; then
        data_dir=multi30k.en-de.mask3
        #data_dir=multi30k.en-fr.mask3
elif [ $which_mask == "mask4" ]; then
        data_dir=multi30k.en-de.mask4
        #data_dir=multi30k.en-fr.mask4
elif [ $which_mask == "maskc" ]; then
        data_dir=multi30k.en-de.maskc
elif [ $which_mask == "mask5" ]; then
        data_dir=multi30k.en-fr.maskc
elif [ $which_mask == "mask6" ]; then
        data_dir=multi30k.en-fr.maskp
elif [ $which_mask == "maskp" ]; then
        data_dir=multi30k.en-de.maskp
else
        echo $which_mask
        exit
fi
fp16=0
if [ $which_data == "catr" ]; then
        image_feat_path=data/catr
elif [ $which_data == "CNN" ]; then
        image_feat_path=data/CNN
	fp16=1
elif [ $which_data == "oscar" ]; then
        image_feat_path=data/oscar
elif [ $which_data == "queryInst" ]; then
        image_feat_path=data/queryInst
elif [ $which_data == "catr_finetune_decoder_out" ]; then
        image_feat_path=data/catr_finetune_decoder_out
elif [ $which_data == "swin_base" ]; then
        image_feat_path=data/swin_base
elif [ $which_data == "beit_base_patch16_384" ]; then
        image_feat_path=data/beit_base_patch16_384
elif [ $which_data == "swin_large" ]; then
        image_feat_path=data/swin_large
elif [ $which_data == "swin_small" ]; then
        image_feat_path=data/swin_small
elif [ $which_data == "swin_tiny" ]; then
        image_feat_path=data/swin_tiny
elif [ $which_data == "detr_resnet101_dc5" ]; then
        image_feat_path=data/detr_resnet101_dc5
elif [ $which_data == "vit_base_patch32_224_in21k" ]; then
        image_feat_path=data/vit_base_patch32_224_in21k
elif [ $which_data == "vit_small_patch16_224" ]; then
        image_feat_path=data/vit_small_patch16_224
elif [ $which_data == "vit_tiny_patch16_224_in21k" ]; then
        image_feat_path=data/vit_tiny_patch16_224_in21k
elif [ $which_data == "detr_resnet101_dc5_catr_vit_tiny_patch16_224_in21k" ]; then
        image_feat_path='data/detr_resnet101_dc5 data/catr data/vit_tiny_patch16_224_in21k'
elif [ $which_data == "vit_base_r50_s16_384" ]; then
        image_feat_path=data/vit_base_r50_s16_384
else
        echo $which_data
        exit
fi

# set device
gpu=0
cpu=

# data set
ensemble=10
batch_size=128
beam=5
src_lang=en

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size 
  --beam $beam 
  --quiet
  --task img_mmt
  --lenpen $length_penalty 
  --output $model_dir/hypo.txt 
  --remove-bpe $use_cpu" 

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi
if [ -n "$image_feat_path" ]; then
cmd=${cmd}" --image-feat-path "${image_feat_path}
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k-en-de/test_origin.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k-en-de/test_origin.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k-en-de/test_origin.coco.de

elif [ $task == "multi30k-fr2de" ] && [ $who == "test" ]; then
	ref=data/multi30k-en-de/test_origin.2016.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k-en-fr/test_origin.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k-en-fr/test_origin.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k-en-fr/test_origin.coco.fr
fi	

hypo=$model_dir/hypo.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log
echo $length_penalty
#python3 get_de_acc.py $hypo $who
#python3 get_fr_acc.py $hypo $who
#python3 get_p_acc.py $hypo $who
#python3 get_en2fr-people_acc.py $hypo $who

