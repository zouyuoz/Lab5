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
        inputs: raw logits
        targets: {0,1}
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BoxLoss(nn.Module):
    def __init__(self, loss_type='ciou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, S, S, A, 4] raw (tx, ty, tw, th)
        target_boxes: [bsz, S, S, A, 4] encoded (tx, ty, w, h) where w,h are normalized to grid (0-1) / image
        anchors: [S, S, A, 2] anchor (w,h) normalized
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        eps = 1e-7

        # Build grid
        gy = torch.arange(grid, device=device).view(1, grid, 1, 1).expand(bsz, grid, grid, num_anchors)
        gx = torch.arange(grid, device=device).view(1, 1, grid, 1).expand(bsz, grid, grid, num_anchors)
        grid_x = gx.float()
        grid_y = gy.float()

        # decode pred
        tx, ty, tw, th = pred_boxes.unbind(-1)
        cx = (tx.sigmoid() + grid_x) / grid
        cy = (ty.sigmoid() + grid_y) / grid
        pw = (tw.clamp(-10, 10).exp()) * anchors[..., 0]
        ph = (th.clamp(-10, 10).exp()) * anchors[..., 1]

        # target (already encoded for center; w,h are normalized absolute)
        t_tx, t_ty, tw_tgt, th_tgt = target_boxes.unbind(-1)
        cx_t = (t_tx.sigmoid() + grid_x) / grid  # if target tx are given as centers-encoded logits; keep consistent
        cy_t = (t_ty.sigmoid() + grid_y) / grid
        # Alternatively, many encodings already give absolute w,h
        pw_t = tw_tgt
        ph_t = th_tgt

        # Convert to corners
        x1 = cx - pw / 2
        y1 = cy - ph / 2
        x2 = cx + pw / 2
        y2 = cy + ph / 2

        x1_t = cx_t - pw_t / 2
        y1_t = cy_t - ph_t / 2
        x2_t = cx_t + pw_t / 2
        y2_t = cy_t + ph_t / 2

        # IoU
        inter_x1 = torch.max(x1, x1_t)
        inter_y1 = torch.max(y1, y1_t)
        inter_x2 = torch.min(x2, x2_t)
        inter_y2 = torch.min(y2, y2_t)
        iw = (inter_x2 - inter_x1).clamp(min=0)
        ih = (inter_y2 - inter_y1).clamp(min=0)
        inter = iw * ih
        area1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area2 = (x2_t - x1_t).clamp(min=0) * (y2_t - y1_t).clamp(min=0)
        union = area1 + area2 - inter + eps
        iou = inter / union

        if self.type == 'giou':
            # enclosing box
            cx1 = torch.min(x1, x1_t)
            cy1 = torch.min(y1, y1_t)
            cx2 = torch.max(x2, x2_t)
            cy2 = torch.max(y2, y2_t)
            c_area = (cx2 - cx1) * (cy2 - cy1) + eps
            giou = iou - (c_area - union) / c_area
            return (1 - giou).squeeze(-1)

        if self.type == 'ciou':
            # center distance
            c_pred_x = (x1 + x2) / 2
            c_pred_y = (y1 + y2) / 2
            c_tgt_x = (x1_t + x2_t) / 2
            c_tgt_y = (y1_t + y2_t) / 2
            rho2 = (c_pred_x - c_tgt_x).pow(2) + (c_pred_y - c_tgt_y).pow(2)

            # enclosing diagonal
            cx1 = torch.min(x1, x1_t)
            cy1 = torch.min(y1, y1_t)
            cx2 = torch.max(x2, x2_t)
            cy2 = torch.max(y2, y2_t)
            c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

            # aspect ratio consistency
            v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(pw_t / (ph_t + eps)) - torch.atan(pw / (ph + eps)), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)

            ciou = iou - (rho2 / c2 + alpha * v)
            return (1 - ciou).squeeze(-1)

        if self.type == 'mse':
            # legacy MSE on encoded params
            target_tx = target_boxes[..., 0]
            target_ty = target_boxes[..., 1]
            target_w = target_boxes[..., 2]
            target_h = target_boxes[..., 3]

            loss_xy = F.mse_loss(tx, target_tx, reduction='none') + F.mse_loss(ty, target_ty, reduction='none')
            target_tw = torch.log(target_w / (anchors[..., 0] + eps) + eps)
            target_th = torch.log(target_h / (anchors[..., 1] + eps) + eps)
            loss_wh = F.mse_loss(tw, target_tw, reduction='none') + F.mse_loss(th, target_th, reduction='none')
            return (loss_xy + loss_wh).squeeze(-1)

        raise NotImplementedError(f"Unknown box loss type: {self.type}")

class YOLOv3Loss_CIoU_Focal(nn.Module):
    """
    Drop-in replacement for YOLOv3Loss:
    - Box: CIoU
    - Objectness: Focal Loss (on logits)
    - Classification: Focal Loss (on logits)
    """
    def __init__(self,
                 lambda_coord=2.0,
                 lambda_obj=1.0,
                 lambda_noobj=0.2,
                 lambda_class=1.0,
                 anchors=None,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.anchors = anchors

        self.box_loss = BoxLoss(loss_type='ciou')
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')

    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [B, S, S, 5 + C] with raw logits
        targets:     list of 3 scales, each [B, S, S, A, 5 + C] encoded
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0

        for scale_idx, pred in enumerate(predictions):
            gt = targets[scale_idx]
            anchors = self.anchors[scale_idx].to(device) if self.anchors is not None else None

            B, S, _, ch = pred.shape
            A = gt.shape[3]  # anchors per cell

            pred = pred.view(B, S, S, A, ch // A) if ch != gt.shape[-1] else pred
            pred_boxes = pred[..., 0:4]     # tx, ty, tw, th (logits/raw)
            pred_obj = pred[..., 4]         # objectness logit
            pred_cls = pred[..., 5:]        # class logits

            target_boxes = gt[..., 0:4]
            target_obj = gt[..., 4]
            target_cls = gt[..., 5:]

            pos_mask = target_obj > 0.5
            neg_mask = ~pos_mask

            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            total_num_pos += num_pos
            total_num_neg += num_neg

            # --- Box loss on positives ---
            if num_pos > 0:
                box_loss_full = self.box_loss(pred_boxes, target_boxes, anchors)
                total_box_loss += box_loss_full[pos_mask].sum()

            # --- Objectness focal loss ---
            if num_pos > 0:
                total_obj_loss_pos += self.focal(pred_obj[pos_mask], target_obj[pos_mask]).sum()
            if num_neg > 0:
                total_obj_loss_neg += self.focal(pred_obj[neg_mask], target_obj[neg_mask]).sum()

            # --- Classification focal loss (positives only) ---
            if num_pos > 0 and target_cls.numel() > 0:
                total_cls_loss += self.focal(pred_cls[pos_mask], target_cls[pos_mask]).sum()

        total_obj_loss = total_obj_loss_pos + total_obj_loss_neg

        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_obj_loss_neg +
            self.lambda_class * total_cls_loss
        )

        return {
            "total": total_loss,
            "box": total_box_loss,
            "obj": total_obj_loss,
            "noobj": total_obj_loss_neg,
            "cls": total_cls_loss,
            "num_pos": torch.tensor(total_num_pos, device=device),
            "num_neg": torch.tensor(total_num_neg, device=device),
        }
