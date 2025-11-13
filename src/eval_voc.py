import sys
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.config import VOC_CLASSES, VOC_IMG_MEAN, VOC_IMG_STD, YOLO_IMG_DIM
import albumentations as A

# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False
):
    """
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    """
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # No predictions made for this class
            ap = 0.0
            print(f"{ap:.5f} AP of class {class_} (no predictions for this class)")
            aps += [ap]
            continue
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.0
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])
        if npos == 0:
            print(f"0.00000 AP of class {class_} (no ground truth instances)")
            aps += [0.0]
            continue
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    union = (
                        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                        + (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                        - inters
                    )
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # bbox has already been used
                        if len(BBGT) == 0:
                            del target[
                                (image_id, class_)
                            ]  # delete things that don't have bbox
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print(f"{ap:.5f} AP of class {class_}")
        aps += [ap]
    print(f"--- MAP: {np.mean(aps):.5f} ---")
    return aps


def evaluate(model, eval_loader):
    """
    Evaluate model using DataLoader with built-in inference method.

    Args:
        model: ODModel instance with inference() method
        eval_loader: DataLoader with encode_target=False

    Returns:
        aps: List of average precision scores per class
    """

    print("---Evaluate model on validation samples---")
    sys.stdout.flush()
    model.eval()
    targets = defaultdict(list)
    preds = defaultdict(list)

    device = next(model.parameters()).device

    # Run inference on all batches
    global_idx = 0
    for images, target_list in tqdm(eval_loader):
        # Move to GPU if available
        images = images.to(device)
        with torch.no_grad():
            detections = model.inference(images, conf_thres=0.3, nms_thres=0.4)

        # Process each image in the batch
        for i in range(len(images)):
            # Use global index as unique image_id
            image_id = global_idx + i

            # Build ground truth from target_list
            # target_list[i] is a list of instances: [[x1,y1,x2,y2,label], ...]
            # Note: boxes are already in pixel coordinates (0-416) after albumentations transform
            for instance in target_list[i]:
                box_pixels = instance[:4]  # [x1, y1, x2, y2] in pixels (0-416)
                label = int(instance[4])
                class_name = VOC_CLASSES[label]
                # Boxes are already in pixel coordinates, no need to scale
                targets[(image_id, class_name)].append(box_pixels)

            # Get image dimensions (all images are resized to YOLO_IMG_DIM)
            # We need original dimensions to convert normalized coords back
            # Since we don't have access to original dims, we'll use YOLO_IMG_DIM
            # Note: This assumes target_list contains boxes in pixel coordinates at YOLO_IMG_DIM
            img_dim = YOLO_IMG_DIM

            # Process detections for this image
            if detections[i] is not None:
                for detection in detections[i]:
                    # Detection format: (x, y, w, h, obj_conf, class_conf, class_pred)
                    x_center = detection[0].item()
                    y_center = detection[1].item()
                    w = detection[2].item()
                    h = detection[3].item()
                    obj_conf = detection[4].item()
                    class_conf = detection[5].item()
                    class_id = int(detection[6].item())

                    # Convert from center format (normalized 0-1) to corner format (pixels)
                    x1 = int((x_center - w / 2) * img_dim)
                    y1 = int((y_center - h / 2) * img_dim)
                    x2 = int((x_center + w / 2) * img_dim)
                    y2 = int((y_center + h / 2) * img_dim)

                    # Clip to image boundaries
                    x1 = max(0, min(x1, img_dim - 1))
                    y1 = max(0, min(y1, img_dim - 1))
                    x2 = max(0, min(x2, img_dim - 1))
                    y2 = max(0, min(y2, img_dim - 1))

                    # Skip degenerate boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Combined confidence score
                    prob = obj_conf * class_conf
                    class_name = VOC_CLASSES[class_id]

                    # Add to predictions
                    preds[class_name].append([image_id, prob, x1, y1, x2, y2])

        global_idx += len(images)

    aps = voc_eval(preds, targets, VOC_CLASSES=VOC_CLASSES)
    return aps

@torch.no_grad()
def evaluate_(model, eval_loader, device=None, conf_thres=0.01, nms_thres=0.90, max_batches=None):
    """
    Quick debug evaluator (NOT mAP). 
    目的：檢查推論分數分佈、是否被閾值/NMS濾光、類別分佈與格式是否正確。
    
    Args:
        model: nn.Module。若有 model.inference(images, conf_thres, nms_thres) 會優先使用。
        eval_loader: DataLoader，輸出 (images, targets_list) 或 (images, _)
        device: 例如 'cuda'。若 None 則不搬運。
        conf_thres: 先設很低（0.01）避免被濾光。
        nms_thres: 先設很寬（0.90）。
        max_batches: 只跑前 N 個 batch，用於快速檢查。
    Returns:
        summary: dict，包含總計與每批資訊，方便你打印或日後比對。
    """
    import torch
    from collections import Counter, defaultdict

    mdl = model
    mdl.eval()
    if device is not None:
        mdl.to(device)

    batch_infos = []
    total_kept = 0
    total_seen = 0
    score_min, score_max = float('inf'), float('-inf')
    class_counter = Counter()

    def _run_infer(m, imgs):
        # 優先用 .inference；否則直接 m(imgs)（假設模型內部已含後處理，若沒有會回 None/空，仍可看出問題）
        if hasattr(m, "inference") and callable(getattr(m, "inference")):
            return m.inference(imgs, conf_thres=0.0, nms_thres=nms_thres)  # 先不過濾，外面再用 conf_thres
        else:
            return m(imgs)

    for bidx, (images, *rest) in enumerate(eval_loader):
        if max_batches is not None and bidx >= max_batches:
            break

        if device is not None:
            images = images.to(device, non_blocking=True)

        dets = _run_infer(mdl, images)  # 期望: list[len=bs], 每個 [N,6或7]
        if not isinstance(dets, (list, tuple)):
            # 若不是 list（例如尚未後處理），直接當作無預測
            dets = [None] * images.shape[0]

        kept_this_batch = 0
        per_class_counter = Counter()
        batch_score_min, batch_score_max = float('inf'), float('-inf')

        for i in range(len(images)):
            d = dets[i]
            if d is None or len(d) == 0:
                continue

            # 解析 score / cls，容錯兩種常見格式：
            # A: [x,y,w,h,obj,cls_conf,cls]  -> score = obj*cls_conf
            # B: [x,y,w,h,score,cls]
            if d.shape[1] >= 7:
                scores = d[:, 4] * d[:, 5]
                cls_ids = d[:, 6].long()
            elif d.shape[1] == 6:
                scores = d[:, 4]
                cls_ids = d[:, 5].long()
            else:
                # 未知格式，跳過但記下
                continue

            # 統計分數門檻後留下的數量
            mask = scores > conf_thres
            kept = int(mask.sum().item())
            kept_this_batch += kept
            total_kept += kept
            total_seen += int(len(scores))

            if len(scores) > 0:
                smin = float(scores.min())
                smax = float(scores.max())
                batch_score_min = min(batch_score_min, smin)
                batch_score_max = max(batch_score_max, smax)
                score_min = min(score_min, smin)
                score_max = max(score_max, smax)

            # 類別統計（只統計門檻後）
            if kept > 0:
                for c in cls_ids[mask].tolist():
                    per_class_counter[c] += 1
                    class_counter[c] += 1

        batch_infos.append({
            "batch_idx": bidx,
            "kept": kept_this_batch,
            "score_min": (None if batch_score_min == float('inf') else batch_score_min),
            "score_max": (None if batch_score_max == float('-inf') else batch_score_max),
            "per_class": dict(per_class_counter)
        })

        print(f"[DBG][batch {bidx}] kept>{conf_thres:.2f} = {kept_this_batch}, "
              f"score_range=[{batch_score_min if batch_score_min!=float('inf') else None}, "
              f"{batch_score_max if batch_score_max!=float('-inf') else None}], "
              f"class_hist={dict(per_class_counter)}")

    summary = {
        "total_batches": len(batch_infos),
        "total_seen_preds": total_seen,
        "total_kept_preds": total_kept,
        "score_range": (None if score_min == float('inf') else score_min,
                        None if score_max == float('-inf') else score_max),
        "class_hist": dict(class_counter),
        "per_batch": batch_infos
    }

    print(f"[DBG][summary] kept>{conf_thres:.2f} = {total_kept}/{total_seen} "
          f"({(100*total_kept/max(total_seen,1)):.2f}%), "
          f"global_score_range={summary['score_range']}, "
          f"class_hist={summary['class_hist']}")
    return summary
