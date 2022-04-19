# Directory

```
.
├── clean_data.py
├── coco_eval.py
├── coco_utils.py
├── data
│   └── quantumml
|       └── quantumml_v1
|  
├── engine.py
├── group_by_aspect_ratio.py
├── predict.py
├── presets.py
├── quantumml_utils.py
├── README.md
├── test.jpg
├── train_maskrcnn.sh
├── train.py
├── transforms.py
└── utils.py
```

# Requirements
```
python 3.7
torch 1.10
opencv-python
pandas
albumentations
tqdm
detectron2
```

# How to run
## Step 1:
Download dataset from CVAT server

## Step 2:
Clean the dataset using scipts: `clean_data.py`

## Step 3:
Train the model:
```
bash train_maskrcnn.sh
```

## Step 4:
Predictions:
```
python predict.py --output-dir logs/maskrcnn_resnet50_fpn_data_v1 --data-path data/quantumml/quantumml_v1/images --rpn-score-thresh 0.3
```
