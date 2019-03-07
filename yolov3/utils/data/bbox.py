import torch


def to_bbox_in_original_image(bboxes, original_image_size, model_input_size):
    """Convert bounding box to the one in the original image

    This function uses an original image size and a model input size to
    transform bounding box coordinates from the ones in the preprocessed
    image to the ones in the original image, because YOLO takes padded
    and resized image as input.

    Args:
        bboxes (torch.tensor): bounding boxes YOLO outputs (shape: Nx4),
            each row is (x0, y0, w, h) where
                (x0, y0): coordinates of top-left corner of bounding box
                (w, h): size of bounding box
        original_image_size (torch.tensor): original image size (shape: 2)
        model_input_size (torch.tensor): model input size (shape: 1)

    Returns:
        (torch.tensor): COCO-style bounding boxes in the original image
            (shape: Nx4)

    """
    w, h = original_image_size
    embedded_image_size = (original_image_size * model_input_size /
                           original_image_size.max()).long()
    offset = (max(embedded_image_size) - min(embedded_image_size)) // 2
    if w < h:
        offset = torch.tensor([offset, 0],
                              dtype=torch.float32,
                              device=bboxes.device)
    else:
        offset = torch.tensor([0, offset],
                              dtype=torch.float32,
                              device=bboxes.device)
    bboxes[:, :2] -= offset
    bboxes *= original_image_size.max() / model_input_size
    return bboxes


def compute_bbox_ious(bboxes1, bboxes2, mode):
    """Compute IoU between bounding boxes

    Args:
        bboxes1 (torch.tensor): bounding boxes
        bboxes2 (torch.tensor: bounding boxes
        mode (str): bounding box type ('xyxy' or 'xhwh')

    Returns:
        (torch.tensor): IoU between given bounding boxes

    """
    if mode == 'xyxy':
        # tl.shape = MxNx2
        tl = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        # br.shape = MxNx2
        br = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        # area1.shape = M
        area1 = torch.prod(bboxes1[:, 2:] - bboxes1[:, :2], dim=1)
        # area2.shape = N
        area2 = torch.prod(bboxes2[:, 2:] - bboxes2[:, :2], dim=1)
    elif mode == 'xywh':
        tl = torch.max((bboxes1[:, None, :2] - bboxes1[:, None, 2:] / 2),
                       (bboxes2[:, :2] - bboxes2[:, 2:] / 2))
        br = torch.min((bboxes1[:, None, :2] + bboxes1[:, None, 2:] / 2),
                       (bboxes2[:, :2] + bboxes2[:, 2:] / 2))
        area1 = torch.prod(bboxes1[:, 2:], dim=1)
        area2 = torch.prod(bboxes2[:, 2:], dim=1)
    else:
        raise ValueError('mode must be either one of xyxy or xywh')
    # is_ok.shape = MxN
    is_ok = (tl < br).float().prod(dim=2)
    # area_intersection.shape = MxN
    area_intersection = torch.prod(br - tl, dim=2) * is_ok
    return area_intersection / (area1[:, None] + area2 - area_intersection)


def nms(bboxes, thresh, scores=None):
    if len(bboxes) == 0:
        return torch.zeros((0, ), dtype=torch.int64)

    if scores is not None:
        order = scores.argsort(descending=True)
        bboxes = bboxes[order]
    areas = torch.prod(bboxes[:, 2:] - bboxes[:, :2], dim=1)

    selected = torch.zeros(len(bboxes), dtype=torch.uint8)
    for index, bbox in enumerate(bboxes):
        tl = torch.max(bbox[:2], bboxes[selected, :2]).float()
        br = torch.min(bbox[2:], bboxes[selected, 2:]).float()
        area_intersection = torch.prod(
            br - tl, dim=1) * (tl < br).all(dim=1).float()
        iou = area_intersection / (
            areas[index] + areas[selected] - area_intersection)
        if (iou >= thresh).any():
            continue
        selected[index] = 1

    selected = selected.nonzero().flatten()
    if scores is not None:
        selected = order[selected]
    return selected.long()
