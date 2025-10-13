import cv2
import numpy as np
# from scipy import stats
import math

from multiprocessing import Process, Manager

def filter_video(from_ind, to_ind):
    for i in range(from_ind, to_ind):
        print(f"{i} out of {to_ind}")
        for j in range(out_height):
            # print(i*out_width+j)
            m_r_a = np.array([], dtype="int64")
            m_g_a = np.array([], dtype="int64")
            m_b_a = np.array([], dtype="int64")
            for k in range((i*out_height)+j, len(pixels), out_height*out_width):
                m_r_a = np.append(m_r_a, pixels[k][0])
                m_g_a = np.append(m_g_a, pixels[k][1])
                m_b_a = np.append(m_b_a, pixels[k][2])
            c_r = np.bincount(m_r_a)
            c_g = np.bincount(m_g_a)
            c_b = np.bincount(m_b_a)
            m_r = np.argmax(c_r)
            m_g = np.argmax(c_g)
            m_b = np.argmax(c_b)
                # m_r, _ = stats.mode(m_r_a)
                # m_g, _ = stats.mode(m_g_a)
                # m_b, _ = stats.mode(m_b_a)
                # m_r = np.median(m_r_a)
                # m_g = np.median(m_g_a)
                # m_b = np.median(m_b_a)
            pixels[(i*out_height)+j] = np.array([m_r,m_g,m_b], dtype=np.uint8)


cap = cv2.VideoCapture("test6.gif")
# cap = cv2.VideoCapture("output.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
y_offset = 0
x_offset = 0
out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, 29.97, frameSize=(out_width, out_height))

pixels = []

while True:
    ret, frame = cap.read()
    # print(frame)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(len(frame[0]))
    for i in range(y_offset,y_offset+out_height):
        for j in range(x_offset,x_offset+out_width):
            pixels.append(frame[i][j])
    # print(frame[0][0])
    # if (len(pixels)>=1000000):
    #     break
    cv2.imshow('video feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


filter_video(0,out_width)

print(len(pixels))

for i in range(len(pixels)//(out_height*out_width)):
    mat = np.array(pixels[out_height*out_width*(i):out_height*out_width*(i+1)])
    mat = np.reshape(mat, (out_height,out_width, 3))
    out.write(mat)

print(mat.shape)


# print(pixels[::10000])