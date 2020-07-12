import cv2
from cv2 import dnn_superres

sr = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('XFILL.jpg')

path = "EDSR_x3.pb"
# path = "FSRCNN_x2.pb"

sr.readModel(path)

sr.setModel("edsr", 3)
# sr.setModel("fsrcnn", 2)

result = sr.upsample(image)

cv2.imwrite("upscaled_x3.png", result)
