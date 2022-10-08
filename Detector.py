from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np

class Detector:
	def __init__(self, case):


		self.cfg = get_cfg()

		if case == "object_detection":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
		elif case == "instance_segmentation":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		elif case == "keypoint_detection":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
		self.cfg.MODEL.DEVICE = "cuda"

		self.predictor = DefaultPredictor(self.cfg)


	def viz_processing(self, image):
		predictions = self.predictor(image)

		viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)

		output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

		return output

	def onImage(self, imagePath):
		image = cv2.imread(imagePath)
		predictions = self.predictor(image)

		viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)

		output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

		cv2.imshow("Result", output.get_image()[:,:,::-1])
		cv2.waitKey(0)

	def onVideo(self, videoPath):
		capture = cv2.VideoCapture(videoPath)

		(is_on, image) = capture.read()

		while is_on:
			output = self.viz_processing(image)
			cv2.imshow("Result", output.get_image()[:,:,::-1])
			if (cv2.waitKey(1) & 0xFF) == ord("q"):
				break
			(is_on, image) = capture.read()

