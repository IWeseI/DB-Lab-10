import glob
import face_recognition
from rtree import index

images = glob.glob("Dataset/**/*.jpg")[:5000]

p = index.Property()
p.dimension = 128
p.buffering_capacity = 16
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('128d_index', properties=p)

for i in range(len(images)):
    image = face_recognition.load_image_file(images[i])
    encoding = tuple(face_recognition.face_encodings(image)[0]) * 2
    idx.insert(i, encoding)
