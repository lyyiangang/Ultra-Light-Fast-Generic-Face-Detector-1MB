#!/usr/bin/env bash
model_root_path="./models/train-time-plate"
train_dataset_dir="../../Dataset/VOC"
num_workers=5
batch_size=200
input_size=128
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --datasets ${train_dataset_dir} \
  --validation_dataset ${train_dataset_dir} \
  --net slim \
  --num_epochs 100 \
  --milestones "50,150" \
  --lr 1e-2 \
  --batch_size ${batch_size} \
  --input_size ${input_size} \
  --checkpoint_folder ${model_root_path} \
  --num_workers ${num_workers} \
  --log_dir ${log_dir} \
  --cuda_index 0 \
  --validation_epochs 1 \
  --resume "./models/train-time-plate/slim-Epoch-0-Loss-0.3717567132365319.pth" \
  2>&1 | tee "$log"
