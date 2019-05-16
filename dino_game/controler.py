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
        self.sct = mss.mss()
        self.mon = {"top": 250, "left": 20, "width": 920, "height": 250}
        # driver.fullscreen_window()
        self.driver.get("chrome://dino")

    def reset(self):
        self.body = self.driver.find_element_by_tag_name('body')
        self.score = 0
        self.body.send_keys(Keys.ENTER)
        # self.driver.execute_script('Runner.instance_.setSpeed(5)')

    def step(self, action):
        # self.driver.execute_script('Runner.instance_.setSpeed(5)')
        image_1 = self.screen_record_efficient()
        reward_1 = Controller.get_reward(image_1)

        if action == 1:  # jump
            self.body.send_keys(Keys.SPACE)

        # cv2.imshow('demo1', image_1)
        # cv2.waitKey(1)

        # time.sleep(0.1)
        #
        # image_2 = self.screen_record_efficient()
        # reward_2 = Controller.get_reward(image_2)

        # cv2.imshow('demo2', image_2)
        # cv2.waitKey(1)

        if reward_1 != self.score:
            # positive action
            self.score = reward_1
            reward = 10
            done = False
        else:
            # negative action
            reward = -100
            done = True

        # print('reward: {}'.format(reward))
        image = cv2.resize(image_1, (460, 125))
        cv2.imshow('demno', image)
        cv2.waitKey(1)
        # image = image/255.0
        return image, reward, done, {}

    def capture_screen(self):
        # file_name = 'observable/{}.png'.format(time.time())
        image = self.driver.get_screenshot_as_png()
        return image

    def screen_record_efficient(self):
        # 920x250 windowed mode
        img = np.asarray(self.sct.grab(self.mon))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    @staticmethod
    def get_reward(original):
        # cropped = original.crop((810, 130, 930, 190))
        cropped_example = original[22:61, 818:915]
        text = pytesseract.image_to_string(cropped_example)
        return text
