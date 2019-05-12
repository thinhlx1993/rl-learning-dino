import os
from PIL import Image
import subprocess
from controler import Controller

import cv2

def ocr(path):
    original = cv2.imread(path)
    cropped_example = original[145:175, 825:914]
    cv2.imwrite('1.png', cropped_example)
    process = subprocess.Popen([r'C:\Program Files\Tesseract-OCR\tesseract.exe', '1.png', '1'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.communicate()

    with open('1.txt', 'r') as handle:
        contents = handle.read()

    return contents


str = ocr('observable/1557585076.2576663.jpg')
print(str)
