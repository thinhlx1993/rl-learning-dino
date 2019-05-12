import cv2
import time
import subprocess
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Controller(object):
    def __init__(self):
        self.score = 0
        self.driver = webdriver.Chrome(executable_path='chromedriver.exe')
        # driver.fullscreen_window()
        self.driver.get("chrome://dino")

    def reset(self):
        ele = self.driver.find_element_by_tag_name('body')
        ele.send_keys(Keys.ENTER)

    def step(self, action, epsilon):
        if action >= epsilon:  # jump
            ele = self.driver.find_element_by_tag_name('body')
            ele.send_keys(Keys.SPACE)

        filename = self.capture_screen()
        reward = Controller.get_reward(filename)
        try:
            reward = int(reward[1:])
            if reward != self.score:
                self.score = reward
                done = False
            else:
                done = True
        except:
            reward = -1
            done = False
        reward = -1
        observalbe = cv2.imread(filename, 1)
        cv2.imshow('date', observalbe)
        cv2.waitKey(1)
        observalbe = observalbe[130:370, 0:928]
        observalbe = cv2.resize(observalbe, (232, 60))
        # observalbe = cv2.cvtColor(observalbe, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('test', observalbe)
        # cv2.waitKey(1)
        observalbe = observalbe/255.
        return observalbe, reward, done, {}

    def capture_screen(self):
        file_name = 'observable/{}.png'.format(time.time())
        self.driver.save_screenshot(file_name)
        return file_name

    @staticmethod
    def get_reward(path):
        original = cv2.imread(path)
        cropped_example = original[130:190, 810:930]
        cv2.imwrite('1.png', cropped_example)
        process = subprocess.Popen([r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe', '1.png', '1'],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.communicate()

        with open('1.txt', 'r') as handle:
            contents = handle.readline()

        return contents
