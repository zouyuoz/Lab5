
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        eps = 1e-7

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
                 lambda_noobj=0.075,
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

        for s, pred in enumerate(predictions):
            gt = targets[s]
            pred, anchors = self._normalize_pred_and_anchor(pred, gt, self.anchors[s], device)

            B,S,_,A,ch = pred.shape
            pred_boxes = pred[...,0:4]
            pred_obj   = pred[...,4]
            pred_cls   = pred[...,5:]

            tgt_boxes = gt[...,0:4]
            tgt_obj   = gt[...,4]
            tgt_cls   = gt[...,5:]

            pos_mask = tgt_obj > 0.5
            neg_mask = ~pos_mask
            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())
            total_pos += num_pos
            total_neg += num_neg

            if num_pos > 0:
                box_loss_map, iou_map = self.box_loss_fn(pred_boxes, tgt_boxes, anchors)
                total_box += box_loss_map[pos_mask].sum()

                if self.use_varifocal:
                    q_target = torch.zeros_like(pred_obj, dtype=pred_obj.dtype)
                    q_target[pos_mask] = iou_map[pos_mask].to(q_target.dtype)
                    vfl = self.vfl(pred_obj, q_target)
                    total_obj_pos += vfl[pos_mask].sum()
                    total_obj_neg += vfl[neg_mask].sum()
                else:
                    total_obj_pos += self.focal(pred_obj[pos_mask], tgt_obj[pos_mask]).sum()
                    total_obj_neg += self.focal(pred_obj[neg_mask], tgt_obj[neg_mask]).sum()

                if tgt_cls.numel() > 0:
                    total_cls += self.focal(pred_cls[pos_mask], tgt_cls[pos_mask]).sum()
            else:
                if self.use_varifocal:
                    q_target = torch.zeros_like(pred_obj)
                    vfl = self.vfl(pred_obj, q_target)
                    total_obj_neg += vfl[neg_mask].sum()
                else:
                    total_obj_neg += self.focal(pred_obj[neg_mask], tgt_obj[neg_mask]).sum()

        pos = max(total_pos, 1)
        box_loss = total_box / pos
        obj_pos  = total_obj_pos / pos
        cls_loss = total_cls / pos
        obj_neg  = self.lambda_noobj * total_obj_neg

        total = (
            self.lambda_coord * box_loss +
            self.lambda_obj   * obj_pos  +
            obj_neg +
            self.lambda_class * cls_loss
        )

        return {
            "total": total,
            "box": box_loss,
            "obj_pos": obj_pos,
            "obj_neg": obj_neg,
            "cls": cls_loss,
            "num_pos": torch.tensor(total_pos, device=device),
            "num_neg": torch.tensor(total_neg, device=device),
        }
