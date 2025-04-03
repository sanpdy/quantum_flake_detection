# QuantumML: Flake Detection for 2D Materials

This repository provides a PyTorch-based pipeline for detecting flakes of 2D materials (e.g., BN, Graphene, MoS2, WTe2) using instance segmentation (Mask R-CNN).

## Features
- COCO-format dataset loading and cleaning utilities
- Training script with distributed training support
- Contrastive-style data augmentation
- Custom evaluation script to compare two model checkpoints
- Mask R-CNN with pre-trained ResNet-50 backbone

## Requirements
```
python 3.7
torch 1.10
opencv-python
pandas
albumentations
tqdm
detectron2
```
## Project Structure
```
quantumml/
├── train.py                   # Main training script
├── train_maskrcnn.sh          # Shell script to run training
├── predict.py                 # Inference script
├── engine.py                  # Training & evaluation engine
├── utils.py                   # Miscellaneous utilities
├── transforms.py              # Data augmentation logic
├── quantumml_utils.py         # Dataset loading and COCO conversion
├── check_dataset.py           # Utility to check for missing files
├── clean_data.py              # Clean/standardize COCO annotations
├── group_by_aspect_ratio.py   # Aspect ratio-based batching
├── coco_eval.py               # COCO-style evaluation
├── coco_utils.py              # COCO utilities
├── logs/                      # Training logs and checkpoints
├── DL_2DMaterials/            # Dataset directory
│   ├── BN/
│   │   ├── train2019/
│   │   ├── val2019/
│   │   └── annotations/
```

## How to Run
### Step 1
Download dataset from CVAT server

### Step 2
Clean the dataset using:
```
python clean_data.py
```

### Step 3
Train the model:
```
bash train_maskrcnn.sh
```

### Step 4
Generate predictions:
```
python predict.py --output-dir logs/maskrcnn_resnet50_fpn_data_v1 --data-path data/quantumml/quantumml_v1/images --rpn-score-thresh 0.3
```

