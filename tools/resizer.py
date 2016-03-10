import os
import sys
from PIL import Image

def resize(folder, fileName, resFolder):
   
    filePath = os.path.join(folder, fileName)
    resPath = os.path.join(resFolder,fileName)
    im = Image.open(filePath)
    w, h  = im.size
    newIm = im.resize((224,224))
    # i am saving a copy, you can overrider orginal, or save to other folder
    newIm.save(resPath)

def bulkResize(imageFolder, resFolder):
    
    imgExts = ["png", "bmp", "jpeg"]
    for path, dirs, files in os.walk(imageFolder):
        for fileName in files:
            ext = fileName[-4:].lower()
            if ext not in imgExts:
                continue
    
            resize(path, fileName, resFolder)

if __name__ == "__main__":
    imageFolder=sys.argv[1]
    resFolder=sys.argv[2]# first arg is path to image folder
    print resFolder
    bulkResize(imageFolder, resFolder)