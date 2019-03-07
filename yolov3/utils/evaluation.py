from pycocotools.cocoeval import COCOeval

stats_names = [
    'AP50_95',
    'AP50',
    'AP75',
    'AP50_95_small',
    'AP50_95_medium',
    'AP50_95_large',
    'AR50_95_maxdets1',
    'AR50_95_maxdets10',
    'AR50_95_maxdets100',
    'AR50_95_small',
    'AR50_95_medium',
    'AR50_95_large',
]


def evaluate(coco, detection_path, image_ids):
    """Evaluate detection result using COCO API

    Args:
        coco (pycocotools.coco.COCO): COCO object
        detection_path (str): JSON file path of predictions
        image_ids (list of int): list of COCO image ids

    Returns:
        (list of float): COCO evaluation results

    """
    detections = coco.loadRes(detection_path)
    coco_eval = COCOeval(cocoGt=coco, cocoDt=detections, iouType='bbox')
    coco_eval.params.image_ids = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
