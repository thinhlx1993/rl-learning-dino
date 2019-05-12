import time
import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Controller(object):
    def __init__(self):
        self.score = 0
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--mute-audio")
        self.driver = webdriver.Chrome(chrome_options=chrome_options, executable_path='chromedriver.exe')
        self.body = None
        # driver.fullscreen_window()
        self.driver.get("chrome://dino")

    def reset(self):
        self.body = self.driver.find_element_by_tag_name('body')
        self.score = 0
        self.body.send_keys(Keys.ENTER)

    def step(self, action):
        if action == 1:  # jump
            self.body.send_keys(Keys.SPACE)

        image = self.screen_record_efficient()
        reward = Controller.get_reward(image)

        try:
            reward = reward.replace('o', '0')
            reward = int(reward[1:])
            if reward != self.score:
                self.score = reward
                done = False
                if action == 0:
                    reward = 2
                else:
                    reward = 1
            else:
                if action == 0:
                    reward = -2
                else:
                    reward = -1
                done = True
        except:
            reward = None
            done = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50, 184))
        image = image/255.0
        return image.flatten(), reward, done, {}

    def capture_screen(self):
        # file_name = 'observable/{}.png'.format(time.time())
        image = self.driver.get_screenshot_as_png()
        return image

    @staticmethod
    def screen_record_efficient():
        # 920x250 windowed mode
        mon = {"top": 250, "left": 20, "width": 920, "height": 250}
        sct = mss.mss()
        img = np.asarray(sct.grab(mon))
        cv2.imshow('demo', img)
        cv2.waitKey(1)
        return img

    @staticmethod
    def get_reward(original):
        # cropped = original.crop((810, 130, 930, 190))
        cropped_example = original[22:61, 818:915]
        text = pytesseract.image_to_string(cropped_example)
        return text
