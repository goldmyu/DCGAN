from PIL import Image
import os, sys

path = "C:/workspace/dcgan datasets/art/sculpture/"
dirs = os.listdir( path )

def resize():
    count = 1
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            try:
                imResize.save('C:/workspace/dcgan datasets/sculpture-resized/'+ str(count) + ' resized.jpg', 'JPEG', quality=90)
            except OSError:
                print("couldnt resize")
            count+=1

resize()