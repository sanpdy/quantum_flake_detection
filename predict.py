from PIL import Image
import numpy as np
import cv2

import torch
import presets
import torchvision
import glob
import os
import tqdm
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Instances


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Inference", add_help=add_help)

    parser.add_argument("--data-path", default="./data/", type=str, help="dataset path")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--num_classes", default=5, type=int, help="Number of classes")
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    return parser


def infer_an_image(image_path, model, transform, args):
    image_name = image_path.split("/")[-1]
    image = Image.open(image_path)
    vis_image = np.array(image)

    image = transform(image, None)[0]
    image = image.unsqueeze(0).cuda()

    with torch.no_grad():
        detections = model(image)

    visualizer = Visualizer(vis_image)
    for detection in detections:
        boxes = detection['boxes']
        scores = detection['scores']
        labels = detection['labels']
        masks = detection['masks']

        nms_idx = torchvision.ops.batched_nms(boxes, scores, labels, 0.45)

        predictions = Instances(vis_image.shape[:2])
        boxes = boxes[nms_idx]
        labels = labels[nms_idx]
        masks = masks[nms_idx]
        scores = scores[nms_idx]

        # import pdb; pdb.set_trace()
        predictions.pred_boxes = boxes.detach().cpu()
        predictions.pred_labels = labels.detach().cpu()
        predictions.pred_scores = scores.detach().cpu()
        predictions.pred_masks = masks[:, 0, ...].detach().cpu() > 0.5


        vis_image = visualizer.draw_instance_predictions(predictions)


        # for box in boxes:
        #     xmin, ymin, xmax, ymax = [int(x) for x in box]
        #     vis_image = cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # import pdb; pdb.set_trace()
    # vis_image = vis_image[:, :, ::-1]
    # cv2.imwrite('{}/{}'.format(args.output_dir, image_name), vis_image)
    vis_image.save('{}/{}'.format(args.output_dir, image_name))


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    num_classes = args.num_classes

    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "box_detections_per_img": 200
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.__dict__[args.model](
        pretrained=False, num_classes=num_classes, **kwargs
    )

    checkpoint = torch.load('{}/checkpoint.pth'.format(args.output_dir))

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    transform = presets.DetectionPresetEval()

    data_path = args.data_path
    if os.path.isfile(data_path):
        infer_an_image(data_path, model, transform, args)
    elif os.path.isdir(data_path):
        image_paths = glob.glob("{}/*.jpg".format(data_path))
        for image_path in tqdm.tqdm(image_paths, total=len(image_paths)):
            infer_an_image(image_path, model, transform, args)
