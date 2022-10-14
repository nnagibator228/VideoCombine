import random
import time
import glob
import cv2
import numpy as np
import multiprocessing
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def render_chunk(file, mask, start, end, number, render_path):
    clip_duration = end - start
    print(f'[DEBUG] number: {number:2} | start: {start:6.2f} | end: {end:6.2f} | duration: {clip_duration:.2f}')
    clip = VideoFileClip(file).subclip(start, end)
    clip = clip.volumex(2)
    clip_mask = VideoFileClip(mask, has_mask=True).resize(clip.size).subclip(start, end)
    video = CompositeVideoClip([clip, clip_mask.set_duration(clip.duration)], size=clip.size)
    video.write_videofile(f"tmp\combined_part{number}.mp4", fps=60)


def copy_file(src):
    cv2.imwrite('output.png', cv2.imread(src), [cv2.IMWRITE_PNG_COMPRESSION])
    print("local image copy created")


def crop_image():
    img = cv2.imread("output.png")
    w = int(img.shape[1])
    h = int(img.shape[0])
    if max(w, h) == w:
        cropped = img[0:h, int((w - h) / 2):int((w - h) / 2) + h]
    elif max(w, h) == h:
        cropped = img[int((h - w) / 2):int((h - w) / 2) + w, 0:w]
    scaled_dim = 750
    cv2.imwrite('output.png', cv2.resize(cropped, (scaled_dim, scaled_dim), interpolation=cv2.INTER_LINEAR),
                [cv2.IMWRITE_PNG_COMPRESSION])
    print("image copy cropped")


def video_combine(length, render_path):
    main = VideoFileClip("img.avi")
    mask = VideoFileClip("mask1.mov", has_mask=True).resize(main.size)
    mask_speedup = mask.speedx(factor=(length / mask.duration))
    video_duration = main.duration
    number = 0
    time_start = time.time()
    args_for_all_processes = []
    for start in range(0, int(video_duration), 30):
        end = start + 30

        if end > video_duration:
            end = video_duration

        number += 1
        print("add process:", number)
        args_for_all_processes.append(("img.avi", "mask1.mov", start, end, number, render_path))
    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(render_chunk, args_for_all_processes)
    print('number of subclips:', number)

    time_end = time.time()

    diff = time_end - time_start
    print(f'time: {diff:.2f}s ({diff // 60:02.0f}:{diff % 60:02.2f})')
    result = concatenate_videoclips([VideoFileClip(f"tmp\combined_part{i+1}.mp4") for i in range(number)])
    result.write_videofile(f"{render_path}\combined.mp4", fps=60)


def cleanup():
    os.remove("output.png")
    for file in glob.glob("tmp/"):
        os.remove(file)
    os.remove("img.avi")
    print("cleanup done")


def test(length, source, render_path):
    supported_extensions = [".png", ".jpg", ".jpeg"]
    source_correct = False
    source_extension = os.path.splitext(source)[1]
    if not (os.path.isfile(source)):
        print("source file not found")
        return 0
    for ext in supported_extensions:
        if source_extension == ext:
            source_correct = True
            break
    if not source_correct:
        print("source file is not correct")
        return 0
    if not (os.path.isdir(render_path)):
        print("render path is not correct or does not exist")
        return 0
    if not (os.path.isfile("mask1.mov")):
        print("no mask file founded")
        return 0
    dur = VideoFileClip("mask1.mov", has_mask=True).duration
    if length > dur:
        print(f"max video length exceeded - duration must be under {dur} sec")
        return 0
    print("input data correct")
    return 1


def image_to_video(time):
    fps = 60
    img = cv2.imread("output.png")
    height, width, layers = img.shape
    size = (width, height)
    frame_count = fps * time
    out = cv2.VideoWriter('img.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(1, frame_count):
        width = int(img.shape[1] * 1.42)
        height = int(img.shape[0] * 1.42)
        dim = (width, height)
        shiftX = random.randint(-1 * (int(img.shape[1] * 0.31 * 0.5) + 1), int(img.shape[1] * 0.31 * 0.5) + 1)
        shiftY = random.randint(-1 * (int(img.shape[0] * 0.31 * 0.5) + 1), int(img.shape[0] * 0.31 * 0.5) + 1)
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        upscaled = cv2.resize(img, dim)
        shifted = cv2.warpAffine(upscaled, M, (upscaled.shape[1], upscaled.shape[0]))
        crop_img = shifted[int(img.shape[0] * 0.31 * 0.5):int(img.shape[0] * 0.31 * 0.5) + img.shape[0],
                   int(img.shape[1] * 0.31 * 0.5):int(img.shape[1] * 0.31 * 0.5) + img.shape[1]]
        out.write(crop_img)
        print(str("%.2f" % float(i / frame_count * 100)) + "% of frames completed")
    out.release()
    print("completed image convertion")


def main(length, source, render_path):
    if test(length, source, render_path):
        time_start = time.time()
        copy_file(source)
        crop_image()
        image_to_video(length)
        video_combine(length, render_path)
        cleanup()
        time_end = time.time()
        diff = time_end - time_start
        print(f'time: {diff:.2f}s ({diff // 60:02.0f}:{diff % 60:02.2f})')


if __name__ == '__main__':
    main(10, 'G:\combiner\source\img4.png', 'G:\combiner\end')
