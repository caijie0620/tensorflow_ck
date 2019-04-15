import numpy as np
import cv2

img = np.load('/home/jie/PycharmProjects/tensorflow_ck/image.npy') # load


print(type(img))
cv2.imshow('image', img)
k = cv2.waitKey(0)
print(k)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()

