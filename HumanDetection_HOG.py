import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
#cap = cv2.VideoCapture(1)

im2 = cv2.imread('images.jfif')
im2 = cv2.imread('people.jfif')

out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480))
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(im2, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(im2, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)

    # Write the output video
    out.write(im2.astype('uint8'))

    cv2.imshow('frame', im2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imwrite('detection_test.jpg',im2)

#cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

