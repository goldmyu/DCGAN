from PIL import Image
import os, sys

input_path = "C:/workspace/dcgan datasets/art/sculpture/"
output_path = 'C:/workspace/dcgan datasets/sculpture-resized/'
dirs = os.listdir(input_path)


def resize():
    count = 1
    for item in dirs:
        if os.path.isfile(input_path + item):
            im = Image.open(input_path + item)
            f, e = os.path.splitext(input_path + item)
            imResize = im.resize((64, 64), Image.ANTIALIAS)
            try:
                imResize.save(output_path + str(count) + ' resized.jpg', 'JPEG',
                              quality=90)
            except OSError:
                print("couldn't resize")
            count += 1


resize()
