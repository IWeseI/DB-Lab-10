import glob
import face_recognition
import random
import pandas as pd
import matplotlib.pyplot as plt
from rtree import index
from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

images = glob.glob("Dataset/**/*.jpg")[:50]

p = index.Property()
p.dimension = 128
p.buffering_capacity = 16
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('128d_index', properties=p)

for i in range(len(images)):
    image = face_recognition.load_image_file(images[i])
    encoding = face_recognition.face_encodings(image)
    if len(encoding) > 0:
        encoding = tuple(encoding[0]) * 2
        idx.insert(i, encoding)

dist = []
for i in range(50):
    while True:
        a = random.randint(0, len(images) - 1)
        image1 = face_recognition.load_image_file(images[a])
        encoding1 = face_recognition.face_encodings(image1)
        if len(encoding1) > 0:
            break
    while True:
        b = random.randint(0, len(images) - 1)
        image2 = face_recognition.load_image_file(images[b])
        encoding2 = face_recognition.face_encodings(image2)
        if len(encoding2) > 0:
            break
    dist.append(face_recognition.face_distance([encoding1[0]], encoding2[0])[0])

suppress_qt_warnings()
commutes = pd.Series(dist)
commutes.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Distribution of random pairs.')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()
