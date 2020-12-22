import cv2
import numpy as np

for i in range (20):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video=cv2.VideoWriter("video"+str(i)+".avi",fourcc,1,(80,80))
    j = 0
    frame = cv2.imread("episode_footage/frame_"+str(i)+str(j)+".png")
    while type(frame) is np.ndarray:
        #cv2.imshow("frame",frame)
        video.write(frame)
        j+=1
        frame = cv2.imread("episode_footage/frame"+str(i)+str(j)+".png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()
