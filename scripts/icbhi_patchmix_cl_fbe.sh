
MODEL="ast_fbe"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}_fbe"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --use_freq_band_embed \
                                        --method patchmix_cl \
                                        --temperature 0.06 \
                                        --proj_dim 768 \
                                        --alpha 1.0 \
                                        --mix_beta 1.0

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_fbe_patchmix_cl_bs8_lr5e-5_ep50_seed1_fbe/best.pth

    done
done
