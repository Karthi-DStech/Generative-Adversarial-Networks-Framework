#!/usr/bin/env bash

# python3 train.py \
# --images_folder '../Datasets/Topographies/raw/FiguresStacked Same Size 4X4' \
# --label_path '../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv' \
# --dataset_name 'biological' \
# --n_epochs 200 \
# --img_size 224 \
# --batch_size 32 \
# --num_workers 4 \
# --train_dis_freq 1 \
# --model_name 'WGANGP' \
# --latent_dim 112 \

python train.py \
--images_folder '../../Dataset/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed' \
--label_path '../../Dataset/biology_data/TopoChip/MacrophageWithClass.csv' \
--dataset_name 'biological' \
--n_epochs 5 \
--img_size 224 \
--batch_size 32 \
--num_workers 4 \
--train_dis_freq 1 \
--model_name 'ACCBlurGAN' \
--n_classes 5 \
--latent_dim 112 \
--embedding_dim 112 \
--in_channels 1 \
--out_channels 1 \