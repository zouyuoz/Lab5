
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.yolo import bbox_iou

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class VarifocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, q):
        with torch.no_grad():
            weight = self.alpha * q + (1.0 - q)
        p = torch.sigmoid(logits)
        diff = (q - p).abs().pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, q, reduction='none')
        loss = diff * weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class BoxCIoULoss(nn.Module):
    def __init__(self, return_iou=True):
        super().__init__()
        self.return_iou = return_iou

    def forward(self, pred_boxes, target_boxes, anchors):
        B, S, _, A, _ = pred_boxes.shape
        device = pred_boxes.device
        dtype = pred_boxes.dtype # <-- 確保定義了 dtype
        eps = 1e-7
    
        if isinstance(anchors, list):
            anchors = torch.tensor(anchors, device=device, dtype=dtype)
            anchors = anchors.view(1, 1, 1, A, 2)
        
        gy = torch.arange(S, device=device).view(1, S, 1, 1).expand(B, S, S, A)
        gx = torch.arange(S, device=device).view(1, 1, S, 1).expand(B, S, S, A)
        grid_x = gx.float()
        grid_y = gy.float()

        tx, ty, tw, th = pred_boxes.unbind(-1)
        cx = (tx.sigmoid() + grid_x) / S
        cy = (ty.sigmoid() + grid_y) / S
        pw = (tw.clamp(-10,10).exp()) * anchors[..., 0]
        ph = (th.clamp(-10,10).exp()) * anchors[..., 1]

        t_tx, t_ty, tw_t, th_t = target_boxes.unbind(-1)
        cx_t = (t_tx.sigmoid() + grid_x) / S
        cy_t = (t_ty.sigmoid() + grid_y) / S
        pw_t = tw_t
        ph_t = th_t

        x1 = cx - pw/2; y1 = cy - ph/2; x2 = cx + pw/2; y2 = cy + ph/2
        x1t = cx_t - pw_t/2; y1t = cy_t - ph_t/2; x2t = cx_t + pw_t/2; y2t = cy_t + ph_t/2

        inter_x1 = torch.max(x1, x1t); inter_y1 = torch.max(y1, y1t)
        inter_x2 = torch.min(x2, x2t); inter_y2 = torch.min(y2, y2t)
        iw = (inter_x2 - inter_x1).clamp(min=0)
        ih = (inter_y2 - inter_y1).clamp(min=0)
        inter = iw * ih
        area1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area2 = (x2t - x1t).clamp(min=0) * (y2t - y1t).clamp(min=0)
        union = area1 + area2 - inter + eps
        iou = inter / union

        cpx = (x1 + x2)/2; cpy = (y1 + y2)/2
        ctx = (x1t + x2t)/2; cty = (y1t + y2t)/2
        rho2 = (cpx - ctx).pow(2) + (cpy - cty).pow(2)

        cx1 = torch.min(x1, x1t); cy1 = torch.min(y1, y1t)
        cx2 = torch.max(x2, x2t); cy2 = torch.max(y2, y2t)
        c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(pw_t/(ph_t+eps)) - torch.atan(pw/(ph+eps)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - (rho2 / c2 + alpha * v)
        loss = (1 - ciou).squeeze(-1)
        if self.return_iou:
            return loss, iou.detach().squeeze(-1)
        return loss

# 假設 bbox_iou 已經從 src.yolo 導入

def get_iou_tensor(pred_boxes_raw, target_boxes_encoded, anchors, grid):
    """
    Decodes raw predictions and encoded targets and computes the IoU tensor.
    pred_boxes_raw: [B, H, W, A, 4] raw (tx, ty, tw, th)
    target_boxes_encoded: [B, H, W, A, 4] encoded (tx, ty, w, h)
    anchors: list of (w, h) for the anchors at this scale (normalized 0-1)
    grid: grid size (e.g., 13)
    """
    bsz, grid_y, grid_x, num_anchors, _ = pred_boxes_raw.size()
    device = pred_boxes_raw.device
    dtype = pred_boxes_raw.dtype
    
    # Reshape anchors for broadcasting [1, 1, 1, A, 2]
    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=device, dtype=torch.float32) 
        anchors = anchors.view(1, 1, 1, num_anchors, 2)
        
    # Create grid indices [1, H, W, 1]
    g_range = torch.arange(grid, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(g_range, g_range, indexing='ij')
    gx = gx.view(1, grid, grid, 1)
    gy = gy.view(1, grid, grid, 1)

    # --- 1. Decode Predicted Boxes to (cx, cy, w, h) normalized (0-1) ---
    pred_tx = pred_boxes_raw[..., 0]
    pred_ty = pred_boxes_raw[..., 1]
    pred_tw = pred_boxes_raw[..., 2]
    pred_th = pred_boxes_raw[..., 3]

    pred_cx = (torch.sigmoid(pred_tx) + gx) / grid
    pred_cy = (torch.sigmoid(pred_ty) + gy) / grid
    
    pred_w = torch.exp(pred_tw.clamp(-10, 10)) * anchors[..., 0] 
    pred_h = torch.exp(pred_th.clamp(-10, 10)) * anchors[..., 1]
    
    # 將預測框塑形成 [N, 4] 格式
    box1 = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1).view(-1, 4) 

    # --- 2. Decode Target Boxes to (cx, cy, w, h) normalized (0-1) ---
    target_tx = target_boxes_encoded[..., 0]
    target_ty = target_boxes_encoded[..., 1]
    target_w = target_boxes_encoded[..., 2]
    target_h = target_boxes_encoded[..., 3]
    
    target_cx = (target_tx + gx) / grid
    target_cy = (target_ty + gy) / grid

    # 將目標框塑形成 [N, 4] 格式
    box2 = torch.stack([target_cx, target_cy, target_w, target_h], dim=-1).view(-1, 4) 
    
    # --- 3. Calculate IoU (Point-wise Calculation) ---
    # 將 [N, 4] 的張量拆解為單一維度 [N] 的張量
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.split(1, dim=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.split(1, dim=1)
    
    # 計算交集 (Intersection)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # 寬高，並確保非負
    iw = torch.clamp(inter_x2 - inter_x1, min=0)
    ih = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = iw * ih
    
    # 計算並集 (Union)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - intersection + 1e-7
    
    # 點對點 IoU
    iou = intersection / union
    
    # 重塑回 [B, H, W, A]
    return iou.view(bsz, grid, grid, num_anchors)

class DetectionLossAdvanced(nn.Module):
    def __init__(self,
                 anchors=None,
                 use_varifocal=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 varifocal_alpha=0.75,
                 varifocal_gamma=2.0,
                 lambda_coord=2.0,
                 lambda_obj=1.0,
                 lambda_noobj=0.2,
                 lambda_class=1.0):
        super().__init__()
        self.anchors = anchors
        self.use_varifocal = use_varifocal
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        self.vfl = VarifocalLoss(alpha=varifocal_alpha, gamma=varifocal_gamma, reduction='none')
        self.box_loss_fn = BoxCIoULoss(return_iou=True)
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    @staticmethod
    def _normalize_pred_and_anchor(pred, gt, anchors, device):
        B = pred.shape[0]
        if pred.dim() == 5 and pred.shape[1] != gt.shape[3]:
            pred = pred.permute(0, 2, 3, 1, 4).contiguous()
        elif pred.dim() == 4:
            S = pred.shape[1]; A = gt.shape[3]; ch = pred.shape[-1] // A
            pred = pred.view(B, S, S, A, ch)
        if anchors is None:
            raise ValueError("anchors is None")
        if not torch.is_tensor(anchors):
            anchors = torch.as_tensor(anchors, dtype=pred.dtype, device=device)
        else:
            anchors = anchors.to(device=device, dtype=pred.dtype)
        if anchors.dim() == 2:
            S = pred.shape[1]; A = pred.shape[3]
            anchors = anchors.view(1,1,A,2).expand(S,S,A,2)
        return pred, anchors

    def forward(self, predictions, targets):
        device = predictions[0].device

        total_box = torch.tensor(0., device=device)
        total_obj_pos = torch.tensor(0., device=device)
        total_obj_neg = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)
        total_pos = 0
        total_neg = 0

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, _ = gt.shape
            
            # Reshape prediction: [B, H, W, 75] -> [B, H, W, 3, 25]
            pred = pred.view(bsz, grid, grid, num_anchors, -1)
            
            # 1. Extract components
            pred_boxes = pred[..., 0:4] # raw: tx, ty, tw, th
            pred_obj = pred[..., 4]     # raw: obj score (logit)
            pred_cls = pred[..., 5:]    # raw: class logits
            
            tgt_boxes = gt[..., 0:4]    # encoded: tx, ty, w, h
            tgt_obj = gt[..., 4]        # target: obj mask (1 or 0)
            tgt_cls = gt[..., 5:]       # target: one-hot class
            
            # Identify Positive and Negative Samples
            pos_mask = tgt_obj.bool()
            neg_mask = (~pos_mask)
            
            # Count samples for normalization
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            total_pos += num_pos
            total_neg += num_neg
            
            # --- Box loss (Coordinate Loss) - Only for positive samples ---
            if num_pos > 0:
                box_loss_full = self.box_loss_fn(
                    pred_boxes, tgt_boxes, anchors
                )
                if isinstance(box_loss_full, (list, tuple)):
                    box_loss_full = box_loss_full[0]
                total_box += box_loss_full[pos_mask].sum()
                
            # --- Objectness Loss (VFL / Focal) ---
            if num_pos > 0 or num_neg > 0:
                if self.use_varifocal:
                    
                    # 1. 計算 IoU 張量 (q_target)
                    iou_tensor = get_iou_tensor(pred_boxes, tgt_boxes, anchors, grid)
                    
                    # 2. 設置 q_target: 負樣本為 0，正樣本為 IoU
                    q_target = torch.zeros_like(pred_obj)
                    if num_pos > 0:
                        q_target[pos_mask] = iou_tensor[pos_mask].to(pred_obj.dtype)
                    
                    # 3. 計算 VFL 損失
                    vfl = self.vfl(pred_obj, q_target)
                    total_obj_pos += vfl[pos_mask].sum()
                    total_obj_neg += vfl[neg_mask].sum()
                    
                else: # Fallback to standard Focal Loss
                    if num_pos > 0:
                        total_obj_pos += self.focal(pred_obj[pos_mask], tgt_obj[pos_mask]).sum()
                    if num_neg > 0:
                        total_obj_neg += self.focal(pred_obj[neg_mask], tgt_obj[neg_mask]).sum()
                        
            # --- Classification Loss - Only for positive samples ---
            if num_pos > 0 and tgt_cls.numel() > 0:
                total_cls += self.focal(pred_cls[pos_mask], tgt_cls[pos_mask]).sum()
        pos = max(total_pos, 1)
        neg = max(total_neg, 1) # <--- 假設 total_neg 在循環中被正確計算

        box_loss = total_box / pos
        obj_pos  = total_obj_pos / pos
        cls_loss = total_cls / pos
        
        # 修正後的 obj_neg 損失：除以總負樣本數 (neg)
        obj_neg  = total_obj_neg / neg

        total = (
            self.lambda_coord * box_loss +
            self.lambda_obj   * obj_pos  +
            self.lambda_noobj * obj_neg +
            self.lambda_class * cls_loss
        )

        return {
            "total": total,
            "box": box_loss,
            "obj_pos": obj_pos,
            "obj_neg": obj_neg,
            "cls": cls_loss,
        }
