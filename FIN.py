from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import cv2
import mediapipe as mp
import imutils
import numpy as np


j = 0
while True:
    #ret, frame = cap.read()
    frame = 'C:/Users/User/Desktop/7777.jpg'
    frame = cv2.imdecode(np.fromfile(frame, dtype=np.uint8), -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯濾波
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    # 開運算去除白色噪點
    kernel = np.ones((9, 9), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN,
                            kernel, iterations=3)
    frame = cv2.Canny(frame, 50, 100)  # 低於50刪除 高於100留下
    contours1, hierarchy = cv2.findContours(frame.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    (contours2, _) = contours.sort_contours(contours1)
    # 若輪廓面積小於100，視為噪音濾除
    contours2 = [i for i in contours2 if cv2.contourArea(i) > 100]
    pixelsPerMetric = None

    for i in contours2:
        # 計算出物品輪廓之外切線框
        box = cv2.minAreaRect(i)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() \
              else cv2.boxPoints(box)
        box = np.array(box, dtype="int")


    def midpoint(point1, point2):
        point = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        return point
    for i in contours2:
        # 計算出物品輪廓之外切線框
        box = cv2.minAreaRect(i)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() \
              else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 左上角座標開始順時針排序，並畫出外切線框
        box = perspective.order_points(box)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

        # 畫書外切線框端點
        for (x, y) in box:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 算出左上和右上端點的中心點、左下和右下端點的中心點
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # 算出左上和左下端點的中心點、右上和右下端點的中心點
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # 計算兩個中心點距離(dA：寬、dB：長)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # 像素值與最左邊物品實際長度比值
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 10.8

        # 計算目標的實際大小
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # 在圖片中標註物體尺寸
        cv2.putText(frame, "{:.1f}cm".format(dimB),
                    (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(frame, "{:.1f}cm".format(dimA),
                    (int(trbrX - 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        for c in contours2:
            # CV2.moments會傳回一系列的moments值，我們只要知道中點X, Y的取得方式是如下進行即可。

            M = cv2.moments(c)

            cX = int(M["m10"] / M["m00"])

            cY = int(M["m01"] / M["m00"])

            # 在中心點畫上黃色實心圓

            cv2.circle(frame, (cX, cY), 10, (1, 227, 254), -1)
            j += 1
    print(j)




    keyName = cv2.waitKey(1)
    cv2.imshow('oxxostudio', frame)

cap.release()
cv2.destroyAllWindows()