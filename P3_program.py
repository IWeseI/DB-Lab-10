import glob
import face_recognition
from queue import PriorityQueue

images = glob.glob("C:/Users/Wese/Downloads/lfw/*")
images = images[:100]
def range_search(image_name, r):
    image = face_recognition.load_image_file(image_name)
    image_encoding = face_recognition.face_encodings(image)[0]
    image_encoding = [image_encoding]
    result = []
    for i in images:
        image_compare = face_recognition.load_image_file(i)
        if len(face_recognition.face_encodings(image_compare))>0:
            image_compare_encoding = face_recognition.face_encodings(image_compare)[0]
            dist = face_recognition.face_distance(image_encoding, image_compare_encoding)
            if dist < r:
                result.append(i)
    return result

def knn_search(image_name, k):
    image = face_recognition.load_image_file(image_name)
    image_encoding = face_recognition.face_encodings(image)[0]
    image_encoding = [image_encoding]
    result = PriorityQueue()
    for i in images:
        image_compare = face_recognition.load_image_file(i)
        if len(face_recognition.face_encodings(image_compare))>0:
            image_compare_encoding = face_recognition.face_encodings(image_compare)[0]
            dist = face_recognition.face_distance(image_encoding, image_compare_encoding)
            result.put((dist, i))
    result_final = []
    for i in range(k):
        result_final.append(result.get())
    return result_final


