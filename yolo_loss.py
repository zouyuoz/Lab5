import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: predictions after sigmoid, shape [N, *]
        targets: ground truth labels (0 or 1), shape [N, *]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4] (raw predictions: tx, ty, tw, th)
        target_boxes: [bsz, grid, grid, anchors, 4] (encoded targets: tx, ty, w, h - w, h are normalized 0-1)
        anchors: list of (w, h) for the anchors at this scale (normalized 0-1)
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype
        eps = 1e-7

        # Reshape anchors to match box dimensions for broadcasting
        anchors = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)

        # coordinate offset for each grid cell
        grid_range = torch.arange(grid, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        # 修正: 將網格索引塑形為 [1, grid, grid, 1] 以正確與 pred_tx ([B, H, W, A]) 廣播
        grid_x = grid_x.view(1, grid, grid, 1) 
        grid_y = grid_y.view(1, grid, grid, 1)

        if self.type == 'giou':
            ##################YOUR CODE HERE##########################
            
            # --- 1. Decode Predicted Boxes (raw tx, ty, tw, th) to (cx, cy, w, h) in normalized image space (0-1) ---
            pred_tx = pred_boxes[..., 0]
            pred_ty = pred_boxes[..., 1]
            pred_tw = pred_boxes[..., 2]
            pred_th = pred_boxes[..., 3]

            # Decoded center coordinates (normalized 0-1)
            pred_cx = (torch.sigmoid(pred_tx) + grid_x) / grid 
            pred_cy = (torch.sigmoid(pred_ty) + grid_y) / grid

            # Decoded width/height (normalized 0-1)
            # Clamp tw, th before exp to prevent overflow
            pred_w = torch.exp(pred_tw.clamp(-10, 10)) * anchors[..., 0] 
            pred_h = torch.exp(pred_th.clamp(-10, 10)) * anchors[..., 1]

            # --- 2. Decode Target Boxes (encoded tx, ty, w, h) to (cx, cy, w, h) in normalized image space (0-1) ---
            target_tx = target_boxes[..., 0]
            target_ty = target_boxes[..., 1]
            # Target w, h are already normalized (0-1)
            target_w = target_boxes[..., 2]
            target_h = target_boxes[..., 3]

            # Target center coordinates (normalized 0-1)
            target_cx = (target_tx + grid_x) / grid
            target_cy = (target_ty + grid_y) / grid

            # Combined decoded boxes (cx, cy, w, h)
            box1 = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
            box2 = torch.stack([target_cx, target_cy, target_w, target_h], dim=-1)

            # --- 3. Convert (cx, cy, w, h) to Corner format (x1, y1, x2, y2) ---
            b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
            b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
            b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
            b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

            # --- 4. Intersection Area ---
            ixmin = torch.max(b1_x1, b2_x1)
            iymin = torch.max(b1_y1, b2_y1)
            ixmax = torch.min(b1_x2, b2_x2)
            iymax = torch.min(b1_y2, b2_y2)
            iw = torch.clamp(ixmax - ixmin, min=0)
            ih = torch.clamp(iymax - iymin, min=0)
            intersection = iw * ih

            # --- 5. Union Area ---
            area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            union = area1 + area2 - intersection + eps
            iou = intersection / union

            # --- 6. Smallest Enclosing Box (C box) ---
            cxmin = torch.min(b1_x1, b2_x1)
            cymin = torch.min(b1_y1, b2_y1)
            cxmax = torch.max(b1_x2, b2_x2)
            cymax = torch.max(b1_y2, b2_y2)
            c_area = (cxmax - cxmin) * (cymax - cymin)

            # --- 7. GIoU Loss ---
            giou = iou - (c_area - union) / (c_area + eps)
            giou_loss = 1.0 - giou
            
            return giou_loss.squeeze(-1)

        elif self.type == 'mse':
            ##################YOUR CODE HERE##########################
            #### MSE box loss ####
            # Target boxes format: [tx, ty, w, h]
            target_tx = target_boxes[..., 0]
            target_ty = target_boxes[..., 1]
            target_w = target_boxes[..., 2]
            target_h = target_boxes[..., 3]

            # Predicted boxes format: [raw_tx, raw_ty, raw_tw, raw_th]
            pred_tx = pred_boxes[..., 0]
            pred_ty = pred_boxes[..., 1]
            pred_tw = pred_boxes[..., 2]
            pred_th = pred_boxes[..., 3]

            # 1. Coordinate (Center) Loss: MSE on tx, ty
            loss_xy = F.mse_loss(pred_tx, target_tx, reduction='none') + \
                      F.mse_loss(pred_ty, target_ty, reduction='none')

            # 2. Size (Width/Height) Loss: MSE on log space (tw, th)
            # Target encoded tw, th: tw = log(w / anchor_w)
            target_tw = torch.log(target_w / anchors[..., 0] + eps)
            target_th = torch.log(target_h / anchors[..., 1] + eps)

            loss_wh = F.mse_loss(pred_tw, target_tw, reduction='none') + \
                      F.mse_loss(pred_th, target_th, reduction='none')

            mse_loss = loss_xy + loss_wh
            
            return mse_loss.squeeze(-1)
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")

class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none') # Focal Loss is provided but BCE is standard for classification
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors  # List of anchor boxes per scale
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 75]
        targets: list of 3 scales, each [batch, grid, grid, 3, 25] (encoded targets)
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0
        batch_size = predictions[0].size(0)

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, _ = gt.shape
            
            # Reshape prediction: [B, H, W, 75] -> [B, H, W, 3, 25]
            pred = pred.view(bsz, grid, grid, num_anchors, -1)
            
            ##################YOUR CODE HERE##########################
            
            # 1. Identify Positive and Negative Samples
            obj_mask = gt[..., 4] # target objectness mask (1 if object assigned, 0 otherwise)
            pos_mask = obj_mask.bool()
            neg_mask = (~pos_mask)
            
            # Count samples for normalization
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            total_num_pos += num_pos
            total_num_neg += num_neg
            
            # 2. Extract components
            pred_boxes = pred[..., 0:4] # raw: tx, ty, tw, th
            pred_obj = pred[..., 4]     # raw: obj score (logit)
            pred_cls = pred[..., 5:]    # raw: class logits (20 classes)
            
            target_boxes = gt[..., 0:4] # encoded: tx, ty, w, h
            target_obj = gt[..., 4]     # target: obj mask (1 or 0)
            target_cls = gt[..., 5:]    # target: one-hot class (20 classes)
            
            # 3. Box Loss (Coordinate Loss) - Only for positive samples (pos_mask == True)
            if num_pos > 0:
                # Calculate loss for all cells/anchors
                box_loss_full = self.box_loss(
                    pred_boxes, target_boxes, anchors
                )
                # Select only the positive samples' loss and sum
                box_loss_pos = box_loss_full[pos_mask].sum()
                total_box_loss += box_loss_pos
            
            # 4. Objectness Loss
            
            # 4a. Positive Objectness Loss (Object Exists) - Only for positive samples
            if num_pos > 0:
                # Target is 1.0
                loss_obj_pos = self.bce_loss(pred_obj[pos_mask], target_obj[pos_mask]).sum()
                total_obj_loss_pos += loss_obj_pos

            # 4b. Negative Objectness Loss (No Object) - Only for negative samples
            if num_neg > 0:
                # Target is 0.0
                # Using BCEWithLogitsLoss on the logits and 0 target
                loss_noobj = self.bce_loss(pred_obj[neg_mask], target_obj[neg_mask]).sum()
                total_obj_loss_neg += loss_noobj

            # 5. Classification Loss - Only for positive samples
            if num_pos > 0:
                # Use BCEWithLogitsLoss on logits and one-hot target
                loss_cls = self.bce_loss(pred_cls[pos_mask], target_cls[pos_mask]).sum()
                total_cls_loss += loss_cls
            
            ##########################################################
            

        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        # Normalize losses
        total_box_loss = total_box_loss / pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom # Positive objectness loss is normalized by total positive samples
        total_cls_loss = total_cls_loss / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom # Negative objectness loss is normalized by total negative samples

        # Combined loss
        
        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_cls_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'noobj': total_noobj_loss,
            'cls': total_cls_loss,
        }
        
        return loss_dict