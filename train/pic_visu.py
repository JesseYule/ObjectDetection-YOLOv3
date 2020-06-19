import cv2
import numpy as np
from PIL import Image
import os

file_name = "../data/raccoon_data/images/raccoon-107.jpg"
assert os.path.isfile(file_name) == True and os.path.isfile("../data/raccoon_data/new_labels.txt")
data = open("../data/raccoon_data/new_labels.txt").readlines()
for i in range(len(data)):
    image_info = data[i].split()
    if image_info[0] == file_name: break

image = cv2.imread(image_info[0])
n_box = len(image_info[1:]) // 5  # xmin, ymin, xmax, ymax, id
for i in range(n_box):
    image = cv2.rectangle(image, (int(float(image_info[1 + i * 5])),
                                  int(float(image_info[2 + i * 5]))),
                          (int(float(image_info[3 + i * 5])),
                           int(float(image_info[4 + i * 5]))), (255, 0, 0), 2)
image = Image.fromarray(np.uint8(image)).resize((int(194 / image.shape[0] * image.shape[1]), 194))
# print("../screenshot/" + file_name.split("/")[-1])
image.save("../screenshot/" + file_name.split("/")[-1])
image.show()
