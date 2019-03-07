import torch


class Detector:
    def __init__(self,
                 model,
                 postprocessor,
                 input_image_size,
                 category_ids=None):
        self.model = model
        self.postprocessor = postprocessor
        self.input_image_size = input_image_size
        self.category_ids = category_ids

    def detect(self, data, image_sizes, image_ids):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)

            results = []
            for output, image_size, image_id in zip(outputs, image_sizes,
                                                    image_ids):
                detections = self.postprocessor(output, image_size,
                                                self.input_image_size)
                detections = detections.cpu().numpy()
                image_id = image_id.cpu().item()

                for detection in detections:
                    bbox = detection[:4]
                    object_conf = detection[4]
                    class_conf = detection[5]
                    class_index = int(detection[6])
                    detection_dict = {
                        'image_id': image_id,
                        'bbox': bbox.tolist(),
                        'score': float(object_conf * class_conf),
                        'segmentation': [],
                    }
                    if self.category_ids is not None:
                        detection_dict.update({
                            'category_id':
                            self.category_ids[int(class_index)]
                        })
                    else:
                        detection_dict.update({
                            'category_index':
                            int(class_index)
                        })
                    results.append(detection_dict)
        return results
