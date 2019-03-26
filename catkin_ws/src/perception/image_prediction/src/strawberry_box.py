#!/usr/bin/env python

'''
  Subscribe image from sensor and use SSD to find the bounding box of strawberry
  and publish the result, image with BBox is also available
'''

import os
import sys
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from image_prediction.msg import bbox, bboxList
from sensor_msgs.msg import Image
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd 

class SSDPrediction(object):
	def __init__(self):
		self.net = build_ssd('test', 300, 2)
		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()
		pkg_path = self.rospack.get_path('image_prediction')
		self.model_name = rospy.get_param("~model_name", "ssd300_strawberry_50000.pth")
		self.confidence_thres = rospy.get_param("~confidence_thres", 0.8) 
								# Only considered as target when confidence greater than
		self.model_path = pkg_path + "/model/" + self.model_name
		self.net.load_weights(self.model_path)
		self.sub_img = rospy.Subscriber("/camera/color/image_raw", Image, \
										self.img_cb)
		self.pub_bb = rospy.Publisher("~bounding_box", bboxList, queue_size = 10)
		self.pub_img_bb = rospy.Publisher("~prediction_res", Image, queue_size = 10)

		rospy.loginfo("Finish loading weights...")

	# 
	def img_cb(self, msg):
		try:
			self.img = self.bridge.imgmsg_to_cv2(msg, "rgb8") # In RGB
		except CvBridgeError as e:
			print(e)
		
		self.vis_img = self.img # Copied one for drawing
		self.img_to_net = self.img_preprocessing() # SSD preprocessing
		var = Variable(self.img_to_net.unsqueeze(0)) # Wrap tensor in Variable
		if torch.cuda.is_available():
			var = var.cuda()
		self.prediction_res = self.net(var) # Forward pass
		self.parse_detection(msg) # Parse prediction result
	# Draw the target that considered as strawberry and publish the pixel of four vertices
	def parse_detection(self, msg):
		pub_msg = bboxList()
		pub_msg.img = msg
		detection = self.prediction_res.data
		scale = torch.Tensor(self.img.shape[1::-1]).repeat(2)
		for i in range(detection.size(1)):
			j = 0
			det = detection[0, i, j, 0].cpu().numpy()
			while detection[0, i, j, 0].cpu().numpy() >= self.confidence_thres:
				score = detection[0, i, j, 0]
				displat_txt = '%d'%(score*100.)
				pt = (detection[0, i, j, 1:]*scale).cpu().numpy()
				pt = [int(p) for p in pt]
				temp = bbox() # Coding: xmin, ymin, xmax, ymax
				temp.bb[0] = pt[0]; temp.bb[1] = pt[1]; temp.bb[2] = pt[2]; temp.bb[3] = pt[3]
				pub_msg.bbox.append(temp)
				pub_msg.num += 1
				# Draw a rectangle
				cv2.rectangle(self.vis_img, (pt[0], pt[1]), \
							(pt[2], pt[3]), (0, 0, 255), 3)
				# Put the text
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(self.vis_img, displat_txt, (pt[0], pt[1]),font, 2, \
                            (255, 255, 255), cv2.LINE_AA)
				j+=1

		#rospy.loginfo("Find {} strawberry".format(pub_msg.num))
		# Publish bbox msg
		self.pub_bb.publish(pub_msg)
		# Publish drawed image
		try:
			self.vis_img = cv2.cvtColor(self.vis_img, cv2.COLOR_BGR2RGB)
			self.pub_img_bb.publish(self.bridge.cv2_to_imgmsg(self.vis_img, "bgr8"))
		except CvBridgeError as e:
			print(e)
	# Preprocessing for SSD
	def img_preprocessing(self):
		res = cv2.resize(self.img, (300, 300)).astype(np.float32)
		res -= (104., 117., 123.)
		res = res.astype(np.float32)
		res = res[:, :, ::-1].copy()
		res = torch.from_numpy(res).permute(2, 0, 1)
		return res

def main(args):
	rospy.init_node("ssd_prediction_node", anonymous = False)
	strawberry = SSDPrediction()
	try:
		rospy.spin()
	except KeyBoardInterrupt:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main(sys.argv)
