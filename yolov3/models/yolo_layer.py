import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov3.utils.data.bbox import compute_bbox_ious


class YOLOLayer(nn.Module):
    def __init__(self, in_channels, all_anchors, anchor_indices,
                 downsample_rate, n_classes, iou_thresh):
        """YOLO layer

        Args:
            in_channels (int): number of input channels
            all_anchors (torch.tensor): all anchors assigned to the
                entire model. We need this because for loss computation,
                each ground-truth bounding box must be assigned to the
                best matching anchor not from anchors assigned to this
                layer but from all the anchors used by the model.
            anchor_indices (torch.tensor): indices of anchors in
                `all_anchors` assigned to this layer
            downsample_rate (int): downsample rate of the input feature
                map to this layer against input image size to the model
            n_classes (int): number of classes
            iou_thresh (float): threshold for IoU not to penalize
                bounding box predictions with sufficiently high IoU
                against ground-truth bounding boxes
        """
        super().__init__()
        self.anchor_indices = nn.Parameter(anchor_indices, requires_grad=False)

        self.downsample_rate = downsample_rate
        self.n_classes = n_classes
        self.iou_thresh = iou_thresh

        all_anchors = all_anchors.float() / downsample_rate
        # anchors assigned to this layer
        self.anchors = nn.Parameter(
            all_anchors[self.anchor_indices], requires_grad=False)

        all_anchors_ = torch.zeros((len(all_anchors), 4), dtype=torch.float32)
        all_anchors_[:, 2:] = all_anchors
        # all anchors considered in the entire model
        self.all_anchors = nn.Parameter(all_anchors_, requires_grad=False)

        self.conv = nn.Conv2d(
            in_channels,
            len(self.anchor_indices) * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x, targets=None):
        # output shape is (n_batches, n_anchors, fsize, fsize, 5 + n_classes)
        outputs = self._forward_and_swap_axis(x)

        if targets is None:
            outputs = self._decode_outputs(outputs, copy=False)
            outputs[..., :4] *= self.downsample_rate
            return outputs.view(outputs.shape[0], -1, outputs.shape[-1])

        targets, target_mask, object_mask, target_scale = self._create_target(
            outputs, targets)

        n_channels = outputs.shape[-1]
        outputs[..., np.r_[0:4, 5:n_channels]] *= target_mask
        outputs[..., 4] *= object_mask
        outputs[..., 2:4] *= target_scale

        targets[..., np.r_[0:4, 5:n_channels]] *= target_mask
        targets[..., 4] *= object_mask
        targets[..., 2:4] *= target_scale

        loss_xy = F.binary_cross_entropy(
            outputs[..., :2],
            targets[..., :2],
            weight=target_scale * target_scale,
            reduction='sum')
        loss_wh = F.mse_loss(
            outputs[..., 2:4], targets[..., 2:4], reduction='sum') / 2
        loss_object = F.binary_cross_entropy(
            outputs[..., 4], targets[..., 4], reduction='sum')
        loss_class = F.binary_cross_entropy(
            outputs[..., 5:], targets[..., 5:], reduction='sum')

        loss = loss_xy + loss_wh + loss_object + loss_class

        return loss, loss_xy, loss_wh, loss_object, loss_class

    def _forward_and_swap_axis(self, x):
        outputs = self.conv(x)

        batch_size, _, fsize = outputs.shape[:3]
        outputs = outputs.view(batch_size, len(self.anchor_indices), -1, fsize,
                               fsize)
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
        n_channels = outputs.shape[-1]

        indices_except_wh = np.r_[:2, 4:n_channels]
        outputs[..., indices_except_wh] = torch.sigmoid(
            outputs[..., indices_except_wh])
        return outputs

    def _decode_outputs(self, outputs, copy=False):
        batch_size, n_anchors, fsize, _, n_channels = outputs.shape
        dtype, device = outputs.dtype, outputs.device

        shape = 1, 1, fsize, fsize
        x_shift = torch.arange(fsize, dtype=dtype, device=device).expand(shape)
        y_shift = torch.arange(
            fsize, dtype=dtype, device=device).view(-1, 1).expand(shape)

        shape = 1, n_anchors, 1, 1
        out_shape = outputs.shape[:4]
        anchor_ws = self.anchors[:, 0].view(shape).expand(out_shape)
        anchor_hs = self.anchors[:, 1].view(shape).expand(out_shape)

        if copy:
            outputs = outputs.clone()
        outputs[..., 0] += x_shift
        outputs[..., 1] += y_shift
        outputs[..., 2] = torch.exp(outputs[..., 2]) * anchor_ws
        outputs[..., 3] = torch.exp(outputs[..., 3]) * anchor_hs

        return outputs

    def _create_target(self, outputs, org_targets):
        decoded_outputs = self._decode_outputs(outputs, copy=True)
        predicted_bboxes = decoded_outputs[..., :4].data

        batch_size, _, fsize, _, n_channels = outputs.shape
        dtype, device = outputs.dtype, outputs.device

        out_shape = outputs.shape[:4]
        target_mask = torch.zeros((*out_shape, n_channels - 1),
                                  dtype=dtype,
                                  device=device)
        object_mask = torch.ones(out_shape, dtype=dtype, device=device)
        target_scale = torch.zeros((*out_shape, 2), dtype=dtype, device=device)
        targets = torch.zeros_like(outputs)

        n_targets_all = (org_targets.sum(dim=2) > 0).sum(dim=1)

        org_targets = org_targets.clone()
        org_targets[:, :, :4] *= fsize
        org_grid_indices = org_targets[:, :, :2].int()

        for image_index in range(batch_size):
            n_targets = int(n_targets_all[image_index])
            if n_targets == 0:
                continue
            gt_grid_indices = org_grid_indices[image_index, :n_targets]

            best_anchor_indices = self._select_best_anchor_indices(
                org_targets[image_index][:n_targets, :4])
            is_iou_high_enough = self._check_if_iou_is_high_enough(
                predicted_bboxes[image_index][:n_targets],
                org_targets[image_index][:n_targets, :4])

            object_mask[image_index, :n_targets] = 1 - is_iou_high_enough

            for index, anchor_index in enumerate(best_anchor_indices):
                if anchor_index not in self.anchor_indices:
                    continue
                anchor_index = (
                    anchor_index == self.anchor_indices).nonzero().flatten()

                xindex, yindex = gt_grid_indices[index]
                object_mask[image_index, anchor_index, yindex, xindex] = 1
                target_mask[image_index, anchor_index, yindex, xindex, :] = 1
                targets[image_index, anchor_index, yindex, xindex, :
                        2] = torch.frac(org_targets[image_index, index, :2])
                targets[image_index, anchor_index, yindex, xindex, 2:
                        4] = torch.log(org_targets[image_index, index, 2:4] /
                                       self.anchors[anchor_index] + 1e-16)
                targets[image_index, anchor_index, yindex, xindex, 4] = 1
                targets[image_index, anchor_index, yindex, xindex, 5 +
                        org_targets[image_index, index, 4].int()] = 1
                target_scale[
                    image_index, anchor_index, yindex, xindex, :] = torch.sqrt(
                        2 - org_targets[image_index, index, 2] *
                        org_targets[image_index, index, 3] / fsize / fsize)
        return targets, target_mask, object_mask, target_scale

    def _select_best_anchor_indices(self, gt_bboxes):
        """Select best matching anchor for each ground-truth bounding box

        Only shape and size are considered, and spatial position is
        irrelevant.
        """
        gt_bboxes = gt_bboxes.clone()
        gt_bboxes[:, :2] = 0
        ious = compute_bbox_ious(gt_bboxes, self.all_anchors, mode='xyxy')
        best_anchor_indices = torch.argmax(ious, dim=1)
        return best_anchor_indices

    def _check_if_iou_is_high_enough(self, preds, gt_bboxes):
        ious = compute_bbox_ious(preds.view(-1, 4), gt_bboxes, mode='xywh')
        highest_iou_for_each_prediction, _ = ious.max(dim=1)
        return (highest_iou_for_each_prediction > self.iou_thresh).view(
            preds.shape[:3])
