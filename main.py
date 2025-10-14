import cv2
import numpy as np
# from scipy import stats
import math
import ffmpeg

from multiprocessing import Process, Manager
import concurrent.futures

import argparse

def stack_videos_vertically(file_names, output_file):
    # Загружаем все видео в ffmpeg.input объекты
    inputs = [ffmpeg.input(name) for name in file_names]

    # Соединяем их вертикально (vstack)
    stacked = ffmpeg.filter_(inputs, 'vstack', inputs=len(inputs))

    # Выводим полученное видео в файл
    ffmpeg.output(stacked, output_file).run()

def writeout(index, data, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output_{index}.avi', fourcc, 29.97, (width, height))

    for i in range(len(data)//(width*height)):
        mat = np.array(data[width*height*(i):width*height*(i+1)])
        mat = np.reshape(mat, (height,width, 3))
        out.write(mat)

def filter_video(index, from_ind, to_ind, pixels, out_width, out_height):
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
    writeout(index, pixels, out_height, to_ind-from_ind)

def filter_video_numpy_cont(index, from_ind, to_ind, pixels, out_width, out_height):
    result = []
    for i in range(pixels.shape[1]):
        # print(f"Done {i} of {pixels.shape[1]}")
        a = []
        for j in range(pixels.shape[2]):
            m_r_a = pixels[:index,i,j,0]
            m_g_a = pixels[:index,i,j,1]
            m_b_a = pixels[:index,i,j,2]
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
            pixel = np.array([m_r,m_g,m_b], dtype=np.uint8)
            a.append(pixel)
        a = np.array(a, dtype=np.uint8)
        result.append(a)
    result = np.array(result, dtype=np.uint8)
    return result


def filter_pixel_numpy_cont(pixels):
    print(f"FUNCTION SHAPE {pixels.shape}")
    m_r_a = pixels[:,0]
    m_g_a = pixels[:,1]
    m_b_a = pixels[:,2]
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
    pixel = np.array([m_r,m_g,m_b], dtype=np.uint8)
    print(pixel)
    exit(0)
    return pixel

def filter_row_numpy_cont(row):
    print(f"FUNCTION SHAPE {row.shape}")
    new_row = []
    for pixel in row:
        m_r_a = pixel[:,0]
        m_g_a = pixel[:,1]
        m_b_a = pixel[:,2]
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
        new_pixel = np.array([m_r,m_g,m_b], dtype=np.uint8)
        new_row.append(new_pixel)
    return new_row

def process_frame(frames):
    return filter_row_numpy_cont(frames)
    # return filter_pixel_numpy_cont(frames)

filename = ""
show_videocap_window: bool = True

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to an input file")
parser.add_argument("-s", "--silent", action="store_true", help="Won't show a video preview window")
args = parser.parse_args()

if args.input:
    filename = args.input

if args.silent:
    show_videocap_window = False

if __name__ == "__main__":

    cap = cv2.VideoCapture(filename)
    # cap = cv2.VideoCapture("output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    x_offset = 0
    y_offset = 0
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi', fourcc, 29.97, (out_width, out_height))

    pixels = []
    cnt = 0
    running = False
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        # print(frame)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(len(frame[0]))

        # cnt = 0
        # for i in range(x_offset,x_offset+out_width, 35):
        #     if running == False:
        #         pixels.append([])
        #     for k in range(i, i+35):
        #         for j in range(y_offset,y_offset+out_height):
        #             pixels[cnt].append(frame[k][j])
        #     cnt += 1
        # running = True

        # print(frame[0][0])
        # if (len(pixels)>=1000000):
        #     break
        if show_videocap_window:
            cv2.imshow('video feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    frames = np.array(frames)
    frames = np.transpose(frames,(1,2,0,3))
    res_frames = []


    print(f"INCOMING SHAPE {frames.shape}")
    # print(frames[0])
    # print("=================")
    # print(frames[0,0])
    # print("=================")
    # print(frames[0,0,0])
    # exit()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_frame, frames))

        # for res in results:
            # print(res)

    res_frames.extend(results)

    res_frames = np.reshape(res_frames, (out_height,out_width,3))
    cv2.imwrite("dedede.jpg", res_frames)
    print(len(results))

    # print(len(pixels))
    # print(cnt)
    # manager = Manager()
    # # for i in range(len(pixels)):
    # #     pixels[i] = manager.list(pixels)
    # # pixels = manager.list(pixels)

    # num_procs = 10
    # chunk = out_width // num_procs
    # processes = []

    # for i in range(num_procs):
    #     start = i * chunk
    #     end = (i + 1) * chunk if i < num_procs - 1 else out_width
    #     p = Process(target=filter_video, args=(i, start, end, pixels[i], out_width, out_height))
    #     # p = Process(target=filter_video, args=(args[i]))
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     p.join()

    # num_files = 9  # например, 3 файла
    # files = [f"output_{i}.avi" for i in range(1, num_files + 1)]
    # output = "vertical_stacked_output.mp4"
    # stack_videos_vertically(files, output)

    # for i in range(len(pixels)//(out_width*out_height)):
    #     mat = np.array(pixels[out_width*out_height*(i):out_width*out_height*(i+1)])
    #     mat = np.reshape(mat, (out_width,out_height, 3))
    #     out.write(mat)

    #     print(mat)


# print(pixels[::10000])