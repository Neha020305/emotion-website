from inference import predict
from PIL import Image
import sys

img = Image.open(sys.argv[1])
print("CNN:   ", predict(img, "cnn"))
print("ResNet:", predict(img, "resnet"))