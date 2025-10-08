import cv2
import numpy as np
# from scipy import stats
import math

from multiprocessing import Process, Manager

def filter_video(from_ind, to_ind, pixels, out_width, out_height):
    for i in range(from_ind, to_ind):
        print(f"{i} out of {out_width}")
        for j in range(out_height):
            # print(i*out_width+j)
            m_r_a = []
            m_g_a = []
            m_b_a = []
            for k in range((i*out_width)+j, len(pixels), out_width*out_height):
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

if __name__ == "__main__":

    cap = cv2.VideoCapture("test2.mp4")
    # cap = cv2.VideoCapture("output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    x_offset = 0
    y_offset = 0
    out_width = 350
    out_height = 350
    out = cv2.VideoWriter('output.avi', fourcc, 29.97, (out_width, out_height))

    pixels = []
    cnt = 0
    running = False

    while True:
        ret, frame = cap.read()
        # print(frame)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(len(frame[0]))
        cnt = 0
        for i in range(x_offset,x_offset+out_width, 35):
            if running == False:
                pixels.append([])
            for k in range(i, i+10):
                for j in range(y_offset,y_offset+out_height):
                    pixels[cnt].append(frame[k][j])
            cnt += 1
        running = True
        # print(frame[0][0])
        # if (len(pixels)>=1000000):
        #     break
        cv2.imshow('video feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


    print(len(pixels))
    print(cnt)
    manager = Manager()
    for i in range(len(pixels)):
        pixels[i] = manager.list(pixels)
    # pixels = manager.list(pixels)

    num_procs = 12
    chunk = out_width // num_procs
    processes = []

    for i in range(num_procs):
        start = i * chunk
        end = (i + 1) * chunk if i < num_procs - 1 else out_width
        p = Process(target=filter_video, args=(start, end, pixels[i], out_width, out_height))
        # p = Process(target=filter_video, args=(args[i]))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    for i in range(len(pixels)//(out_width*out_height)):
        mat = np.array(pixels[out_width*out_height*(i):out_width*out_height*(i+1)])
        mat = np.reshape(mat, (out_width,out_height, 3))
        out.write(mat)

        print(mat)


# print(pixels[::10000])