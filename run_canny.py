# Use this script to run and test your edge detector. For each of the two
# provided sample images, it should create and write the following images
# to disk, using the best parameters (per image) you were able to find:
#
# 1) The smoothed horizontal and vertical gradients (2 images).
# 2) The gradient magnitude image.
# 3) The gradient magnitude image after suppression.
# 4) The results of your full edge detection function.
#
# The image naming convention isn't important- this script exists for you
# to test and experiment with your code, and to figure out what the best
# parameters are. As far as automated testing is concerned, only the
# five functions in canny.py must adhere to a specific interface. 

# TODO: Implement me!
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import sqrt, pi
import seaborn as sns; sns.set_theme()
from canny import *
import os

os.makedirs("output/", exist_ok=True)

img_mandrill = cv2.imread('example_images/mandrill.jpg', 1)
gray_mandrill = cv2.cvtColor(img_mandrill, cv2.COLOR_BGR2GRAY)
edgemap_mandrill = cannyEdgeDetection(gray_mandrill, sigma=1, tL=0.2, tH=0.8)
cv2.imwrite("output/mandrill_edges.jpeg", edgemap_mandrill*255)

img_cs = cv2.imread('example_images/csbldg.jpg', 1)
gray_cs = cv2.cvtColor(img_cs, cv2.COLOR_BGR2GRAY)
edgemap_cs = cannyEdgeDetection(gray_cs, sigma=1, tL=0.2, tH=0.8)
ax6 = sns.heatmap(edgemap_cs)
cv2.imwrite("output/csbldg_edges.jpeg", edgemap_cs*255)