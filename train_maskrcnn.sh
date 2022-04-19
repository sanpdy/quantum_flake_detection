set -e
set -x

torchrun --nproc_per_node=1 train.py \
    --data-path data/quantumml/ \
    --model maskrcnn_resnet50_fpn \
    --lr 0.0002 \
    --epochs 1000 \
    --lr-scheduler cosineannealinglr \
    --output-dir ./logs/maskrcnn_resnet50_fpn_data_v1
