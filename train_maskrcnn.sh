set -e
set -x

torchrun --nproc_per_node=1 train.py \
    --data-path /home/sankalp/quant_flakes/quantumml/DL_2DMaterials/BN/ \
    --model maskrcnn_resnet50_fpn \
    --lr 0.0002 \
    --epochs 300 \
    --lr-scheduler cosineannealinglr \
    --output-dir ./neurips_logs/ \
    --data-augmentation contrastive \
    --pretrained-backbone  