import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

PATH = os.getcwd()

# This class represents the instance sefmentation model Faster RCNN
class Segmentor:
    
    def __init__(self):
        self._cfg = get_cfg()
        self._enggine = 'gpu' if torch.cuda.is_available() else 'cpu'
        self._predictor = self.makePredictor()
    
    # Initializes the model and configureation to return predictor
    def makePredictor(self):
        self._cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
        self._cfg.MODEL.DEVICE = self._enggine
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self._cfg.MODEL.WEIGHTS = os.path.join(PATH, 'app/models/model_final.pth')
        return DefaultPredictor(self._cfg)
    
    def predict(self, image):
        return self._predictor(image)