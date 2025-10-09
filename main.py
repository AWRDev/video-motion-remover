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
            m_r_a = []
            m_g_a = []
            m_b_a = []
            for k in range((i*out_height)+j, len(pixels), out_height*out_width):
                # print(f"{(i*100+1)*(j+1)+j}")
                # print(k)
                pass
                # print(f"DO: {pixels[k]}")
                m_r_a.append(pixels[k][0])
                # print(m_r)
                m_g_a.append(pixels[k][1])
                m_b_a.append(pixels[k][2])
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
                pixels[k] = np.array([m_r,m_g,m_b], dtype=np.uint8)
                # pixels[k] = temp
                # print(f"POSLE: {pixels[k]}")


cap = cv2.VideoCapture("test4.gif")
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

    # num_procs = 1
    # chunk = out_width // num_procs
    # processes = []

    # for i in range(num_procs):
    #     start = i * chunk
    #     end = (i + 1) * chunk if i < num_procs - 1 else out_width
    #     p = Process(target=filter_video, args=(start, end, pixels, out_width, out_height))
    #     # p = Process(target=filter_video, args=(args[i]))
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     p.join()

for i in range(len(pixels)//(out_height*out_width)):
    mat = np.array(pixels[out_height*out_width*(i):out_height*out_width*(i+1)])
    mat = np.reshape(mat, (out_height,out_width, 3))
    out.write(mat)

print(mat.shape)


# print(pixels[::10000])