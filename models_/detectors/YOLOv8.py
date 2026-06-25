import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8:
    """
    YOLOv8 person detector wrapper for simple-HRNet.

    Uses the Ultralytics YOLO API (pip install ultralytics).
    Produces detections in the same (x1, y1, x2, y2, conf, cls_conf, cls_pred)
    format that SimpleHRNet expects, so no changes are needed in SimpleHRNet.py.
    """

    def __init__(self,
                 model_def='yolov8m',
                 conf_thres=0.3,
                 device=torch.device('cpu')):
        """
        Args:
            model_def (str): YOLOv8 model name or path to a custom .pt file.
                Built-in sizes: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x.
                For a nano (fastest) model use 'yolov8n'.
                Default: 'yolov8m'
            conf_thres (float): minimum confidence threshold for detections.
                Default: 0.3
            device (torch.device): inference device.
                Default: torch.device('cpu')
        """
        self.model_def = model_def
        self.conf_thres = conf_thres
        self.device = device

        # Ultralytics YOLO handles model download automatically
        self.model = YOLO(self.model_def if self.model_def.endswith('.pt')
                          else self.model_def + '.pt')

        # Move model to the requested device
        device_str = str(self.device)
        self.model.to(device_str)

    def predict_single(self, image, color_mode='BGR'):
        """
        Run inference on a single image.

        Args:
            image (np.ndarray): HxWxC image array.
            color_mode (str): 'BGR' (OpenCV default) or 'RGB'.

        Returns:
            torch.Tensor: shape (N, 7) — x1 y1 x2 y2 conf cls_conf cls_pred,
                          or None if no persons detected.
        """
        # Ultralytics accepts BGR or RGB; we pass source='bgr' flag via the
        # predict API so we do not need to convert manually.
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            classes=[0],          # class 0 = person in COCO
            verbose=False,
            device=str(self.device),
            # Ultralytics expects BGR by default when input is np.ndarray
            # (same as OpenCV), so no conversion required.
        )

        result = results[0]  # single image → first (only) result
        boxes_data = result.boxes

        if boxes_data is None or len(boxes_data) == 0:
            return None

        # xyxy coordinates on CPU
        xyxy = boxes_data.xyxy.cpu()          # (N, 4)  x1 y1 x2 y2
        conf = boxes_data.conf.cpu()          # (N,)
        cls  = boxes_data.cls.cpu()           # (N,)

        # Build a (N, 7) tensor that matches the YOLOv5 interface:
        #   x1  y1  x2  y2  obj_conf  cls_conf  cls_id
        # For YOLOv8 there is no separate objectness score; we duplicate conf.
        detections = torch.cat(
            [xyxy,
             conf.unsqueeze(1),
             conf.unsqueeze(1),   # cls_conf == conf (YOLOv8 has no separate obj score)
             cls.unsqueeze(1)],
            dim=1
        )

        return detections

    def predict(self, images, color_mode='BGR'):
        raise NotImplementedError(
            "Batch prediction is not currently supported for YOLOv8. "
            "Use predict_single per frame."
        )
