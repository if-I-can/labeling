import os
import gc
import time
import cv2
import json
import torch
import numpy as np
import argparse
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo
from demo_lazy import get_parser, setup_cfg
import ape
from ape.model_zoo import get_config_file

def setup_APE(version):
    logger = setup_logger(name="ape")
    running_device = "cuda" if torch.cuda.is_available() else "cpu"
    running_device = "cpu"
    if version == 'D':
        config_file = "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
        init_checkpoint = "/home/wch/3.8t_1/Workspace/wch/PROJECT/APE/weights/D/model_final.pth"
    else:
        raise ValueError(f"Unsupported APE version: {version}")

    args = get_parser().parse_args([])
    args.config_file = get_config_file(config_file)
    args.confidence_threshold = 0.01
    args.opts = [
        f"train.init_checkpoint='{init_checkpoint}'",
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=True",
    ]
    if running_device == "cpu":
        args.opts.append("model.model_language.dtype='float32'")
    
    cfg = setup_cfg(args)
    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device
    ape.modeling.text.eva02_clip.factory._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    demo = VisualizationDemo(cfg, args=args, parallel=False)
    # gc.collect()
    # torch.cuda.empty_cache()

    # gc.collect()
    # torch.cuda.empty_cache()
    demo.predictor.model.to(running_device)
    # demo.predictor.model.half()
    gc.collect()
    torch.cuda.empty_cache()
    return demo, logger

def calculate_iou(box1, box2):
    """计算两个边界框的IoU（交并比）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def filter_similar_boxes(boxes, scores, iou_threshold=0.5):
    """过滤相似的边界框，保留得分最高的"""
    indices = np.argsort(scores)[::-1]
    keep = []
    for i in indices:
        keep_box = True
        for j in keep:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                keep_box = False
                break
        if keep_box:
            keep.append(i)
    return np.array(keep)

def calculate_overlap(box1, box2):
    """计算两个边界框的重叠面积比例"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    overlap_ratio1 = intersection / area1 if area1 > 0 else 0
    overlap_ratio2 = intersection / area2 if area2 > 0 else 0

    return overlap_ratio1, overlap_ratio2

def filter_large_boxes(boxes, overlap_threshold=0.7):
    """过滤包含其他框的大框"""
    keep = np.ones(len(boxes), dtype=bool)
    for i, box1 in enumerate(boxes):
        if not keep[i]:
            continue
        for j, box2 in enumerate(boxes):
            if i != j and keep[j]:
                overlap1, overlap2 = calculate_overlap(box1, box2)
                if overlap2 > overlap_threshold:
                    keep[i] = False
                    break
    return np.where(keep)[0]

def convert_to_yolo_format(box, image_width, image_height):
    """Convert bounding box to YOLO format."""
    x_center = (box[0] + box[2]) / 2 / image_width
    y_center = (box[1] + box[3]) / 2 / image_height
    width = (box[2] - box[0]) / image_width
    height = (box[3] - box[1]) / image_height
    return [x_center, y_center, width, height]

def expand_box(box, expand_ratio=0.15):
    """Expand the bounding box by a given ratio."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    x_expand = width * expand_ratio
    y_expand = height * expand_ratio
    return [
        max(0, box[0] - x_expand),
        max(0, box[1] - y_expand),
        box[2] + x_expand,
        box[3] + y_expand
    ]

def run_APE(demo, logger, input_image_path, input_text, score_threshold, output_folder, json_data):
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    original_image = cv2.imread(input_image_path)
    input_image = read_image(input_image_path, format="BGR")
    image_height, image_width = original_image.shape[:2]
    start_time = time.time()
    predictions, _, _, _ = demo.run_on_image(
        input_image,
        text_prompt=input_text,
        mask_prompt=None,
        with_box=True,
        with_mask=False,
        with_sseg=False,
    )

    logger.info(
        "{} in {:.2f}s".format(
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    instances = predictions["instances"].to(demo.cpu_device)
    boxes = instances.pred_boxes.tensor.numpy()
    print(boxes)
    scores = instances.scores.numpy()

    # Apply filters
    keep_similar = filter_similar_boxes(boxes, scores)
    boxes = boxes[keep_similar]
    scores = scores[keep_similar]

    keep_large = filter_large_boxes(boxes)
    boxes = boxes[keep_large]
    # Create a folder for this image's crops
    image_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    crop_folder = os.path.join(output_folder, "crops", image_filename)
    os.makedirs(crop_folder, exist_ok=True)

    # Prepare YOLO format data and crop seeds
    yolo_boxes = []
    for i, box in enumerate(boxes.astype(int)):
        # Convert to YOLO format
        yolo_box = convert_to_yolo_format(box, image_width, image_height)
        yolo_boxes.append(yolo_box)

        # Crop and save seed image
        expanded_box = expand_box(box)
        crop = original_image[int(expanded_box[1]):int(expanded_box[3]), int(expanded_box[0]):int(expanded_box[2])]
        crop_filename = f"seed_{i}.jpg"
        cv2.imwrite(os.path.join(crop_folder, crop_filename), crop)

    # Add YOLO format data to JSON
    json_data.append({
        "image": os.path.basename(input_image_path),
        "boxes": yolo_boxes
    })
    print("====",len(yolo_boxes))
    # Draw bounding boxes on the original image
    for box in boxes:
        cv2.rectangle(original_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    gc.collect()
    torch.cuda.empty_cache()
    return original_image

def process_folder(input_folder, output_folder, input_text, score_threshold, ape_version):
    demo, logger = setup_APE(ape_version)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "crops"), exist_ok=True)

    json_data = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"output_{filename}")

            try:
                output_image = run_APE(demo, logger, input_path, input_text, score_threshold, output_folder, json_data)
                
                # Save output image
                cv2.imwrite(output_image_path, output_image)

                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Save JSON data
    with open(os.path.join(output_folder, "yolo_data.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    print("All images processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with APE.")
    parser.add_argument("--input_folder", required=True, help="Input folder path")
    parser.add_argument("--output_folder", required=True, help="Output folder path")
    parser.add_argument("--input_text", required=True, help="Input text for APE")
    parser.add_argument("--score_threshold", type=float, required=True, help="Score threshold")
    parser.add_argument("--ape_version", required=True, help="APE version")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.input_text, args.score_threshold, args.ape_version)