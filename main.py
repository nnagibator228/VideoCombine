import random

import cv2
import numpy as np
from moviepy.editor import *

# creates a local copy of source file
def copy_file(src):
    cv2.imwrite('output.png', cv2.imread(src), [cv2.IMWRITE_PNG_COMPRESSION])
    print("local image copy created")

# crops image to fit square 750x750px
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

# combines generated video with mask
def video_combine(length, render_path):
    main = VideoFileClip("img.avi")
    mask = VideoFileClip("mask1.mov", has_mask=True).resize(main.size)
    mask_speedup = mask.speedx(factor=(length / mask.duration))
    video = CompositeVideoClip([main, mask_speedup.set_duration(main.duration)], size=main.size)
    video.write_videofile(f"{render_path}\combined.mp4", fps=60)

# cleanup remaining files
def cleanup():
    os.remove("img.avi")
    os.remove("output.png")
    print("cleanup done")

# checks correctness of the input data 
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

# creates randomly shift frames sequence
def image_to_video(time):
    img_array = []
    fps = 60
    img = cv2.imread("output.png")
    for i in range(1, fps * time):
        img_array.append(img)
    height, width, layers = img.shape
    size = (width, height)

    out = cv2.VideoWriter('img.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        frame = img_array[i]
        width = int(frame.shape[1] * 1.31)
        height = int(frame.shape[0] * 1.31)
        dim = (width, height)
        shiftX = random.randint(-1 * (int(frame.shape[1] * 0.31 * 0.5) + 1), int(frame.shape[1] * 0.31 * 0.5) + 1)
        shiftY = random.randint(-1 * (int(frame.shape[0] * 0.31 * 0.5) + 1), int(frame.shape[0] * 0.31 * 0.5) + 1)
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        upscaled = cv2.resize(frame, dim)
        shifted = cv2.warpAffine(upscaled, M, (upscaled.shape[1], upscaled.shape[0]))
        crop_img = shifted[int(img.shape[0] * 0.31 * 0.5):int(img.shape[0] * 0.31 * 0.5) + img.shape[0],
                   int(img.shape[1] * 0.31 * 0.5):int(img.shape[1] * 0.31 * 0.5) + img.shape[1]]
        out.write(crop_img)
        print(str("%.2f" % float(i / len(img_array) * 100)) + "% of frames completed")
    out.release()
    print("completed image convertion")

# main function that combines all other functions
def main(length, source, render_path):
    if test(length, source, render_path):
        copy_file(source)
        crop_image()
        image_to_video(length)
        video_combine(length, render_path)
        cleanup()

# requested arguments: video length (in secs, under 150s), source image to be used, folder to export result
if __name__ == '__main__':
    main(90, 'material.png', 'D:\test1\result')
