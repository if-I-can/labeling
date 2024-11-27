import json
import os
import numpy as np

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_ground_truth(gt_dir):
    """加载真实标签"""
    gt_data = {}
    for filename in os.listdir(gt_dir):
        if filename.endswith(".txt"):
            img_id = filename.replace(".txt", "")
            with open(os.path.join(gt_dir, filename), "r") as f:
                gt_data[img_id] = [list(map(float, line.strip().split())) for line in f]
    return gt_data

def compute_precision_recall(tp, fp, total_gt):
    """计算精度和召回率"""
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (total_gt + 1e-10)
    
    return precisions, recalls

def compute_average_precision(tp, fp, total_gt):
    """计算单类别的平均精度 AP"""
    precisions, recalls = compute_precision_recall(tp, fp, total_gt)
    
    # 修正 Precision-Recall 曲线
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # 计算 AP
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap, precisions, recalls

def compute_map(predictions, ground_truths, iou_thresholds):
    """计算 mAP"""
    all_aps = []
    all_precisions = []
    all_recalls = []
    
    for iou_threshold in iou_thresholds:
        tp_list = []
        fp_list = []
        total_gt = 0
        
        for img_id, preds in predictions.items():
            img_id = img_id.replace('.jpg', '')  # 去掉.jpg后缀
            print(img_id)
            gt_boxes = ground_truths.get(img_id, [])
            total_gt += len(gt_boxes)
            gt_matched = [False] * len(gt_boxes)
            preds_sorted = preds  # 不考虑 score，直接使用预测框
            
            for pred_box in preds_sorted:
                best_iou = 0
                best_gt_idx = -1
   
                for i, gt_box in enumerate(gt_boxes):
                    if not gt_matched[i]:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)
        
        # 计算单个 IoU 阈值下的 AP、精度和召回率
        ap, precisions, recalls = compute_average_precision(tp_list, fp_list, total_gt)
        all_aps.append(ap)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        
        # 打印单个 IoU 阈值下的详细信息
        print(f"IoU Threshold {iou_threshold:.2f}:")
        print(f"  Average Precision (AP): {ap:.4f}")
        if len(precisions) > 0 and len(recalls) > 0:
            print(f"  Final Precision: {precisions[-1]:.4f}")
            print(f"  Final Recall: {recalls[-1]:.4f}")
        print()
    
    # 计算 mAP
    map_value = np.mean(all_aps)
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
    
    return map_value, all_aps, all_precisions, all_recalls

# 加载预测 JSON 文件
with open("/home/zsl/label_everything/dataset1结果/yolo_data.json", "r") as f:
    predictions_list = json.load(f)

# 将预测数据转换为字典格式
predictions = {item["image"].replace(".png", ""): item["boxes"] for item in predictions_list}
# print(predictions.keys())

# 加载真实标签
gt_dir = "/home/zsl/label_everything/dataset1/gt"  # 替换为真实标签目录
ground_truths = load_ground_truth(gt_dir)


# 计算 mAP
iou_thresholds = [0.5]
map_value, all_aps, all_precisions, all_recalls = compute_map(predictions, ground_truths, iou_thresholds)