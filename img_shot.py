import cv2
import os


def mkdir(path):
    path = path.strip()
    path= path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        print(path + ' already exist')
        return False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
count = 0
sample_amount = 20
img_list = []
dst = "Testing_Demo_Data/rotate"
mkdir(dst)
while count < sample_amount:
    __, frame = cap.read()
    path = "{}/{}.png".format(dst, count)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img = frame
        img_list.append(img)
        cv2.imshow("taken", img)
        cv2.waitKey(0)
        cv2.imwrite(path, img)
        count += 1
cap.release()
cv2.destroyAllWindows()