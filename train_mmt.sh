#! /usr/bin/bash
set -e

device=0,1
task=multi30k-en2de
image_feat=vit_tiny_patch16_384
mask_data=mask0
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'multi30k-en2de' ]; then
	src_lang=en
	tgt_lang=de
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-de
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-de.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-de.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-de.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-de.mask4
        else
		echo 'error, mask_data:' $mask_data 'not found'
		exit
	fi
elif [ $task == 'multi30k-en2fr' ]; then
	src_lang=en
	tgt_lang=fr
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-fr
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-fr.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-fr.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-fr.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-fr.mask4
        else
		echo 'error, mask_data:' $mask_data 'not found'
		exit
	fi
fi

share_embedding=1
share_decoder_input_output_embed=0
criterion=label_smoothed_cross_entropy #_mmt
fp16=1
seed=1
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
weight_decay=0.0
keep_last_epochs=10
patience=10
max_update=8000
dropout=0.3

arch=transformer_sa_TTop #sa_TTop
sa_text_dropout=0.1
sa_attention_dropout=0.2

which_mask=${array[n1-1]}
which_drop=${array[n1]}
which_data=${array[n1-2]}

if [ $image_feat == "vit_tiny_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=192
else
        echo 
        exit
fi

if [ $which_mask == "mask0" ]; then
        data_dir=multi30k.en-de
elif [ $which_mask == "mask1" ]; then
        data_dir=multi30k.en-de.mask1
elif [ $which_mask == "mask2" ]; then
        data_dir=multi30k.en-de.mask2
elif [ $which_mask == "mask3" ]; then
        data_dir=multi30k.en-de.mask3
elif [ $which_mask == "mask4" ]; then
        data_dir=multi30k.en-de.mask4
elif [ $which_mask == "maskc" ]; then
        data_dir=multi30k.en-de.maskc
else
        echo $which_mask
        exit
fi

if [ $which_drop == "imgdrop0.1" ]; then
        sa_image_dropout=0.1
elif [ $which_drop == "imgdrop0.2" ]; then
        sa_image_dropout=0.2
elif [ $which_drop == "imgdrop0.3" ]; then
        sa_image_dropout=0.3
else
        echo $which_drop
        exit
fi

cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="fairseq-train data-bin/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --task img_mmt
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --criterion $criterion --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --find-unused-parameters
  --no-progress-bar
  --seed $seed
  --weight-decay $weight_decay
  --log-interval 10000
  --ddp-backend no_c10d
  --save-dir $save_dir
  --tensorboard-logdir $tensorboard_logdir 
  --keep-last-epochs $keep_last_epochs"

adam_betas="'(0.9, 0.98)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $best_checkpoint_metric == "bleu" ]; then
cmd=${cmd}" --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --patience 8 "
else
cmd=${cmd}" --patience 10 "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ -n "$sa_image_dropout" ]; then
cmd=${cmd}" --sa-image-dropout "${sa_image_dropout}
fi
if [ -n "$sa_text_dropout" ]; then
cmd=${cmd}" --sa-text-dropout "${sa_text_dropout}
fi
if [ -n "$sa_attention_dropout" ]; then
cmd=${cmd}" --sa-attention-dropout "${sa_attention_dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$image_feat_path" ]; then
cmd=${cmd}" --image-feat-path "${image_feat_path}
fi
if [ -n "$image_feat_dim" ]; then
cmd=${cmd}" --image-feat-dim "${image_feat_dim}
fi
if [ -n "$encoder_embed_dim" ]; then
cmd=${cmd}" --encoder-embed-dim "${encoder_embed_dim}
fi
if [ -n "$decoder_embed_dim" ]; then
cmd=${cmd}" --decoder-embed-dim "${decoder_embed_dim}
fi
if [ -n "$encoder_ffn_embed_dim" ]; then
cmd=${cmd}" --encoder-ffn-embed-dim "${encoder_ffn_embed_dim}
fi
if [ -n "$decoder_ffn_embed_dim" ]; then
cmd=${cmd}" --decoder-ffn-embed-dim "${decoder_ffn_embed_dim}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
