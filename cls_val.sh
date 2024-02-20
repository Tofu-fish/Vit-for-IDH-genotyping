# Set the path to save checkpoints
OUTPUT_DIR='output/linprobe'
# path to imagenet-1k set
DATA_PATH='./MAE_data/IDH'
# path to pretrain model
MODEL_PATH='output/linprobe_IDH_vote/checkpoint-best.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python run_class_val.py \
    --device cuda \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --data_set custom \
    --resume ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 24 \
    --nb_classes 2 \
    --mixup 0 \
    --cutmix 0 \
    --num_workers 8 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 400 \
    --dist_eval
