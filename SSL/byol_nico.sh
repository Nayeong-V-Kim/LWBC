python main_byol_pretraining.py \
  -a resnet18 \
  --lr 0.015 \
  --batch_size 256 --epochs 200 \
  --gpus 0 \
  --save_freq 10 \
  --input_size 224 \
  --byol_hidden_dim 512 \
  --byol_dim 128 \
  --aug_plus \
  --data ../nico_dataset \
  --save_dir './nico_checkpoints'