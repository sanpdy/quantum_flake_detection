from PIL import Image
import numpy as np
import cv2
import torch
import presets
import torchvision
import glob
import os
import tqdm
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Inference", add_help=add_help)

    parser.add_argument("--data-path", default="./data/", type=str, help="dataset path")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--num_classes", default=5, type=int, help="Number of classes")
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone")
    return parser


def infer_an_image(image_path, model, transform, args):
    image_name = os.path.basename(image_path)
    # Load the original image (make sure it is in RGB mode)
    orig_image = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_image)

    # Prepare image for the model using the evaluation transform.
    transformed_image = transform(orig_image, None)[0]
    transformed_image = transformed_image.unsqueeze(0).cuda()

    with torch.no_grad():
        detections = model(transformed_image)
    
    # Use the original image for visualization.
    visualizer = Visualizer(orig_np)
    count = 0
    for detection in detections:
        boxes = detection['boxes']
        scores = detection['scores']
        labels = detection['labels']
        masks = detection['masks']

        nms_idx = torchvision.ops.batched_nms(boxes, scores, labels, 0.45)

        predictions = Instances(orig_np.shape[:2])
        predictions.pred_boxes = boxes[nms_idx].detach().cpu()
        predictions.pred_labels = labels[nms_idx].detach().cpu()
        predictions.pred_scores = scores[nms_idx].detach().cpu()
        if len(detection['boxes']) > 0:
            count += 1
        # Here we threshold the mask; adjust the threshold as needed.
        predictions.pred_masks = masks[nms_idx][:, 0, ...].detach().cpu() > 0.3

        vis_result = visualizer.draw_instance_predictions(predictions)
    if count == 0:
        return

    print(count)
    
    # Get the visualized image as a numpy array and convert it to a PIL image if needed.
    vis_np = vis_result.get_image() if hasattr(vis_result, "get_image") else vis_result
    if not isinstance(vis_np, Image.Image):
        vis_image = Image.fromarray(vis_np)
    else:
        vis_image = vis_np

    # Create a new image that concatenates the original and the visualization side by side.
    combined_width = orig_image.width + vis_image.width
    combined_height = max(orig_image.height, vis_image.height)
    combined_image = Image.new("RGB", (combined_width, combined_height))
    combined_image.paste(orig_image, (0, 0))
    combined_image.paste(vis_image, (orig_image.width, 0))

    # Save the combined image
    combined_image.save(os.path.join(args.output_dir, image_name))


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
        pretrained=True, num_classes=num_classes, **kwargs
    )

    checkpoint = torch.load('/home/sankalp/quant_flakes/quantumml/logs/maskrcnn_resnet50_fpn_data_v1/eighty_twenty.pth')
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
