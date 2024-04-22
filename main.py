import numpy as np
import cv2
##Вариант №3
def task_1():
    img = cv2.imread('images/variant-3.jpeg')
    # изменяем размер изображения на более удобный
    img = cv2.resize(img, dsize=(500,300),fx= 30, fy=10)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    cv2.imshow('image', img_HSV)
#
#
#
def task_2_3():#второе и третье задание.(Работают только на довольно близком расстоянии) 
    video = cv2.VideoCapture(0)
    image= cv2.imread('ref-point.jpg',0)
    imagesize = image.shape[:2]
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = 0.5
        result = cv2.matchTemplate(gray, image, cv2.TM_CCOEFF_NORMED)
        locations = []
        for y,x in zip(*np.where(result >= threshold)):
            locations.append((x, y))
        for loc in locations:
            cv2.rectangle(frame, (loc[0], loc[1]), (loc[0]+imagesize[0],loc[1]+imagesize[1]), (0, 255, 255), 3)
        #Реализация третьего задания
        framesize = frame.shape
        rectsize = (200,200)
        cordsOfRect = (framesize[1]//2 - rectsize[0]//2, framesize[0]//2- rectsize[1]//2)
        cv2.rectangle(frame,(cordsOfRect[0], cordsOfRect[1]), (cordsOfRect[0] + rectsize[0], cordsOfRect[1]+rectsize[1]), (255,255 , 255), 3)
        for loc in locations:
            if (cordsOfRect[0] < loc[0]+imagesize[0]//2 <cordsOfRect[0] + rectsize[0]) and (cordsOfRect[1] < loc[1]+imagesize[1]//2 <cordsOfRect[1] + rectsize[1]):
                cv2.rectangle(frame,(cordsOfRect[0], cordsOfRect[1]), (cordsOfRect[0] + rectsize[0], cordsOfRect[1]+rectsize[1]), (255,0, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video.release()
#
#
#
def task4():#дополнительное задание 
    video = cv2.VideoCapture(0)
    image= cv2.imread('ref-point.jpg',0)
    imagesize = image.shape[:2]
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = 0.5
        result = cv2.matchTemplate(gray, image, cv2.TM_CCOEFF_NORMED)
        locations = []
        for y,x in zip(*np.where(result >= threshold)):
            locations.append((x, y))
        for loc in locations:
            cv2.rectangle(frame, (loc[0], loc[1]), (loc[0]+imagesize[0],loc[1]+imagesize[1]), (0, 255, 255), 3)
        #
        #
        #
        flyimage = cv2.imread('fly64.png')
        flyshape = flyimage.shape[:2]
        for loc in locations:
            x,y = loc
            x += imagesize[1]//2-flyshape[1]//2
            y += imagesize[0]//2-flyshape[0]//2
            try:
                frame[y:y+ flyshape[0],x:x+ flyshape[1]] = flyimage[:,:]
            except:
                print('Oops, something go wrong')
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()
#
#
#

if __name__ == '__main__':
    task_1()
#    task_2_3()
#    task4()


cv2.waitKey(0)
cv2.destroyAllWindows()