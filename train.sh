#!/bin/bash
python main.py \
	      --model CMMPNet \
        --batch_size 4 \
        --gpu_ids 0 \
        --epochs 30 \
        --dataset "BJRoad" \
        --sat_dir "dataset/BJRoad/train_val/image" \
	      --mask_dir "dataset/BJRoad//train_val/mask" \
        --gps_dir "dataset/BJRoad/train_val/gps" \
        --test_sat_dir "dataset/BJRoad/test/image" \
        --test_mask_dir "dataset/BJRoad/test/mask" \
        --test_gps_dir "dataset/BJRoad/test/gps" \
        --weight_save_dir 'save_model/' \
        --lr 1e-4 \
        --down_scale 'true'