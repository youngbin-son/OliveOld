import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print("Cannot recieve frame (stream end?). Exiting...")
        break
    cv2.imshow("Original", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        cv2.imwrite('./imgs/makeup.jpg',frame)
    if key & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

detector = dlib.get_frontal_face_detector()
shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def align_faces(img):       #얼굴 정렬하는 함수
    dets = detector(img)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)
    return faces

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

def preprocess(img):
    return img / 127.5 - 1
def deprecess(img):
    return (img+1) / 2

img1 = dlib.load_rgb_image('./imgs/makeup.jpg')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./UI/군인.PNG')
img2_faces = align_faces(img2)

# fig, axes = plt.subplots(1,2,figsize=(8,5))
# axes[0].imshow(img1_faces[0])
# axes[1].imshow(img2_faces[0])
# plt.show()

src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprecess(output[0])

# plt.savefig('./imgs/makeuped.jpg',output_img)
output_img_uint8 = (output_img * 255).astype(np.uint8)  # 이미지를 0~255 범위의 정수로 변환
output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)  # matplotlib과 cv2의 색 공간 차이를 해결하기 위해 변환

cv2.imwrite('./imgs/makeuped.jpg', output_img_bgr)  # 이미지 저장

fig, axes = plt.subplots(1,3,figsize=(16,10), num = "make up")
axes[0].imshow(img1_faces[0])
axes[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
axes[1].imshow(img2_faces[0])
axes[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
axes[2].imshow(output_img)
axes[2].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.show()