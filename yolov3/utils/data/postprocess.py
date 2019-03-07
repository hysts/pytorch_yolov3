import torch

import yolov3.utils.data.bbox


class PostProcessor:
    def __init__(self, conf_thresh, nms_thresh):
        """Initialize postprocessor

        Args:
            conf_thresh (float): confidence threshold
            nms_thresh (float): threshold for NMS

        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, detections, original_image_size, model_input_size):
        """Post-process

        - Remove detections with low confidences
        - Assign each bounding box with the most confident class
        - Apply class-wise NMS
        - Represent bounding box as a pair of coordinates of top-left
            corner and its size
        - Remove padding and resize to the scale in the original image

        Args:
            detections (torch.tensor): detection results
                (shape: N x (5 + n_classes)) whose row is (bbox,
                object_conf, class_preds) where
                    bbox: bounding box (shape: 4)
                        bounding box is represented as (xc, yc, w, h) where
                            (xc, yc): coordinates of bounding box center
                            (w, h): size of bounding box
                    object_conf: object_confidence (shape: 1)
                    class_preds: class-wise predictions (shape: n_classes)
            original_image_size (torch.tensor): size of original image
                (shape: 2)
            model_input_size (torch.tensor): size of model input (shape: 1)

        Returns:
            (torch.tensor): detection results in the original image
                (shape: M x 7), whose row is (bbox, object_conf,
                class_conf, class_index) where
                    bbox: bounding box (shape: 4)
                        bounding box is represented as (x0, y0, w, h)
                        where
                            (x0, y0): coordinates of top-left corner of
                                the bounding box
                            (w, h): size of bounding box
                    object_conf: object confidence (shape: 1)
                    class_conf: class confidence (shape: 1)
                    class_index: class index (shape: 1)

        """
        detections = self._remove_detections_with_low_confidence(detections)
        if len(detections) == 0:
            return torch.tensor([])
        detections = self._assign_most_confident_class_to_bbox(detections)
        class_indices = detections[:, 6].long().flatten()

        outputs = []
        for class_index in class_indices.unique():
            class_detections = detections[class_indices == class_index]
            class_detections = self._apply_nms(class_detections)
            outputs.append(class_detections)
        outputs = torch.cat(outputs)

        # Change bbox representation from (xc, yc, w, h) to (x0, y0, w, h)
        outputs[:, :2] -= outputs[:, 2:4] / 2

        outputs[:, :4] = yolov3.utils.data.bbox.to_bbox_in_original_image(
            outputs[:, :4], original_image_size, model_input_size)

        return outputs

    def _remove_detections_with_low_confidence(self, detections):
        """Remove detections with low confidence

        Args:
            detections (torch.tensor): detection results
                (shape: N x (5 + n_classes)) whose row is (bbox, object_conf,
                class_preds) where
                    bbox: bounding box (shape: 4)
                    object_conf: object_confidence (shape: 1)
                    class_preds: class-wise predictions (shape: n_classes)

        Returns:
            (torch.tensor): selected detection results
                (shape: M x (5 + n_classes))

        """
        object_confs = detections[:, 4]
        class_preds = detections[:, 5:]
        class_confs, _ = torch.max(class_preds, dim=1)
        return detections[object_confs * class_confs >= self.conf_thresh]

    @staticmethod
    def _assign_most_confident_class_to_bbox(detections):
        """Assign each bounding box with the most confident class

        Args:
            detections (torch.tensor): detection results
                (shape: N x (5 + n_classes)) whose row is (bbox,
                object_conf, class_preds) where
                    bbox: bounding box (shape: 4)
                    object_conf: object_confidence (shape: 1)
                    class_preds: class-wise predictions (shape: n_classes)

        Returns:
            (torch.tensor): modified detection results (shape: N x 7)
                whose row is (bbox, object_conf, class_conf, class_index)
                where
                    bbox: bounding box (shape: 4)
                    object_conf: object confidence (shape: 1)
                    class_conf: class confidence (shape: 1)
                    class_index: class index (shape: 1)

        """
        class_preds = detections[:, 5:]
        confs, indices = torch.max(class_preds, dim=1, keepdim=True)
        detections = torch.cat([detections[:, :5], confs,
                                indices.float()],
                               dim=1)
        return detections

    def _apply_nms(self, class_detections):
        """Apply NMS to bounding boxes

        Args:
            class_detections (torch.tensor): detection results for a class
                (shape: Nx7), whose row is (bbox, object_conf, class_conf,
                class_index) where
                    bbox: bounding box (shape: 4)
                        bounding box is represented as (xc, yc, w, h) where
                            (xc, yc): coordinates of bounding box center
                            (w, h): size of bounding box
                    object_conf: object confidence (shape: 1)
                    class_conf: class confidence (shape: 1)
                    class_index: class index (shape: 1)

        Returns:
            (torch.tensor): suppressed detection results

        """
        # convert bbox format from (xc, yc, w, h) to (x0, y0, x1, y1)
        bboxes = class_detections[:, :4].clone()
        bboxes[:, :2] -= bboxes[:, 2:] / 2
        bboxes[:, 2:] += bboxes[:, :2]
        scores = class_detections[:, 4] * class_detections[:, 5]
        indices = yolov3.utils.data.bbox.nms(
            bboxes, self.nms_thresh, scores=scores)
        return class_detections[indices]
