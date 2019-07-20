import time
import cv2
import numpy as np
import pygame
import random

from PIL import Image
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model, Sequential


class Controller(object):
    def __init__(self):
        self.pygame = pygame
        self.pygame.init()
        self.display_width = 800
        self.display_height = 600

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)

        self.car_width = 50
        self.car_height = 100
        # start_music = pygame.mixer.Sound("Hurry_Up.mp3")
        self.clock = pygame.time.Clock()

        self.carImg = pygame.image.load("car_racing/car1.png")  # load the car image
        self.car2Img = pygame.image.load("car_racing/car2.png")
        self.bgImg = pygame.image.load("car_racing/back2.jpg")
        self.crash_img = pygame.image.load("car_racing/crash.png")
        self.svs = pygame.image.load("car_racing/svs.png")

        self.gameDisplay = self.pygame.display.set_mode((self.display_width, self.display_height))

        # high score
        self.count = 0

        # game data
        self.bg_x1 = (self.display_width / 2) - (360 / 2)
        self.bg_x2 = (self.display_width / 2) - (360 / 2)
        self.bg_y1 = 0
        self.bg_y2 = -600
        self.bg_speed = 10
        self.bg_speed_change = 0
        self.car_x = ((self.display_width / 2) - (self.car_width / 2))
        self.car_y = (self.display_height - self.car_height)
        self.car_x_change = 0
        self.road_start_x = (self.display_width / 2) - 112
        self.road_end_x = (self.display_width / 2) + 112

        self.thing_startx = random.randrange(self.road_start_x, self.road_end_x - self.car_width)
        self.thing_starty = -600
        self.thingw = 50
        self.thingh = 100
        self.thing_speed = 15

        # init models
        self.autoencoder, self.encoder = self.build_autoencoder()
        self.actor = self.build_actor(4, 3)

    def reset(self):
        """
        Reset env when game is done
        :return:
        """
        # game data
        self.bg_x1 = (self.display_width / 2) - (360 / 2)
        self.bg_x2 = (self.display_width / 2) - (360 / 2)
        self.bg_y1 = 0
        self.bg_y2 = -600
        self.bg_speed = 10
        self.bg_speed_change = 0
        self.car_x = ((self.display_width / 2) - (self.car_width / 2))
        self.car_y = (self.display_height - self.car_height)
        self.car_x_change = 0
        self.road_start_x = (self.display_width / 2) - 112
        self.road_end_x = (self.display_width / 2) + 112

        self.thing_startx = random.randrange(self.road_start_x, self.road_end_x - self.car_width)
        self.thing_starty = -600
        self.thingw = 50
        self.thingh = 100
        self.thing_speed = 15
        self.count = 0

    def build_autoencoder(self):
        input_img = Input(shape=(286, 110, 3))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2, (2, 2), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Conv2D(2, (2, 2), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (2, 2), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (4, 4), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        encoder = Model(input_img, encoded)
        return autoencoder, encoder

    def build_actor(self, action_space, state_size):
        # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
        model = Sequential()
        model.add(Conv2D(16, (2, 2), activation='relu', input_shape=(18, 7, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(3, activation='linear'))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        # for layer in inception_model.layers:
        #     layer.trainable = False

        model.compile(loss='mse', optimizer='adam')
        # model.load_weights(model_path)
        return model

    def _train_autoencoder(self, x_train):
        self.autoencoder.fit(
            x_train, x_train,
            epochs=1,
            verbose=1
        )

    def step(self, action, show_capture=True, ai_control=False):
        done = False
        car_x_change = 0
        reward = 2 if action == 0 else 1
        # manual control

        if not ai_control:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        car_x_change = -10
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        car_x_change = 10
                        action = 1

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        car_x_change = 0
                        action = 0
        else:
            # auto control
            car_x_change = 0
            if action == 1:
                car_x_change = 10  # turn right
            elif action == 2:
                car_x_change = -10  # turn left

        self.car_x += car_x_change

        if self.car_x > self.road_end_x - self.car_width:
            self.crash(self.car_x, self.car_y)
            done = True
            reward = -5
        if self.car_x < self.road_start_x:
            self.crash(self.car_x - self.car_width, self.car_y)
            done = True
            reward = -5

        if self.car_y < self.thing_starty + self.thingh:
            if self.thing_startx <= self.car_x <= self.thing_startx + self.thingw:
                self.crash(self.car_x - 25, self.car_y - self.car_height / 2)
                done = True
                reward = -10

            if self.thing_startx <= self.car_x + self.car_width <= self.thing_startx + self.thingw:
                self.crash(self.car_x, self.car_y - self.car_height / 2)
                done = True
                reward = -10

        self.gameDisplay.fill(self.green)  # display white background

        self.gameDisplay.blit(self.bgImg, (self.bg_x1, self.bg_y1))
        self.gameDisplay.blit(self.bgImg, (self.bg_x2, self.bg_y2))
        self.gameDisplay.blit(self.svs, (10, (self.display_height / 2) - 100))
        self.gameDisplay.blit(self.svs, (self.display_width - 200 - 10, (self.display_height / 2) - 100))
        self.car(self.car_x, self.car_y)  # display car
        self.draw_things(self.thing_startx, self.thing_starty, self.car2Img)
        self.highscore(self.count)
        self.count += 1

        self.thing_starty += self.thing_speed

        if self.thing_starty > self.display_height:
            self.thing_startx = random.randrange(self.road_start_x, self.road_end_x - self.car_width)
            self.thing_starty = -200

        self.bg_y1 += self.bg_speed
        self.bg_y2 += self.bg_speed

        if self.bg_y1 >= self.display_height:
            self.bg_y1 = -600

        if self.bg_y2 >= self.display_height:
            self.bg_y2 = -600

        self.pygame.display.update()  # update the screen
        self.clock.tick(24)  # frame per sec
        data = pygame.image.tostring(self.gameDisplay, 'RGB')
        img = Image.frombytes('RGB', (800, 600), data)
        observable = img.crop((290, 0, 510, 570))
        basewidth = 110
        baseheight = 286
        # wpercent = (basewidth / float(observable.size[0]))
        # hsize = int((float(observable.size[1]) * float(wpercent)))
        observable = observable.resize((basewidth, baseheight), Image.ANTIALIAS)
        observable = np.asarray(observable)
        observable = observable / 255.
        observable = np.expand_dims(observable, axis=0)
        self._train_autoencoder(observable)
        observable = self.encoder.predict(observable)
        # observable = np.array([self.car_x, self.car_y, self.thing_startx, self.thing_starty])
        return observable, reward, done, action

    def create_opencv_image_from_stringio(self, img_stream, cv2_img_flag=0):
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.float16)
        return cv2.imdecode(img_array, cv2_img_flag)

    @staticmethod
    def calculate_reward(done, action):
        reward = 1
        if done:
            reward = -10

        return reward

    def intro(self):
        # pygame.mixr.Sound.play(start_music)
        intro = True
        menu1_x = 200
        menu1_y = 400
        menu2_x = 500
        menu2_y = 400
        menu_width = 100
        menu_height = 50
        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            pygame.display.set_icon(self.carImg)

            pygame.draw.rect(self.gameDisplay, self.black, (200, 400, 100, 50))
            pygame.draw.rect(self.gameDisplay, self.black, (500, 400, 100, 50))

            self.gameDisplay.fill(self.white)
            self.message_display("CAR RACING", 100, self.display_width / 2, self.display_height / 2)
            self.gameDisplay.blit(self.svs, ((self.display_width / 2) - 100, 10))
            pygame.draw.rect(self.gameDisplay, self.green, (200, 400, 100, 50))
            pygame.draw.rect(self.gameDisplay, self.red, (500, 400, 100, 50))

            mouse = pygame.mouse.get_pos()
            click = pygame.mouse.get_pressed()

            if menu1_x < mouse[0] < menu1_x + menu_width and menu1_y < mouse[1] < menu1_y + menu_height:
                pygame.draw.rect(self.gameDisplay, self.blue, (200, 400, 100, 50))
                if click[0] == 1:
                    intro = False
            if menu2_x < mouse[0] < menu2_x + menu_width and menu2_y < mouse[1] < menu2_y + menu_height:
                pygame.draw.rect(self.gameDisplay, self.blue, (500, 400, 100, 50))
                if click[0] == 1:
                    pygame.quit()
                    quit()

            self.message_display("Go", 40, menu1_x + menu_width / 2, menu1_y + menu_height / 2)
            self.message_display("Exit", 40, menu2_x + menu_width / 2, menu2_y + menu_height / 2)

            pygame.display.update()
            self.clock.tick(50)

    def highscore(self, count):
        font = self.pygame.font.SysFont(None, 20)
        text = font.render("Score : " + str(count), True, self.black)
        self.gameDisplay.blit(text, (0, 0))

    def draw_things(self, thingx, thingy, thing):
        self.gameDisplay.blit(thing, (thingx, thingy))

    def car(self, x, y):
        self.gameDisplay.blit(self.carImg, (x, y))

    def text_objects(self, text, font):
        textSurface = font.render(text, True, self.black)
        return textSurface, textSurface.get_rect()

    def message_display(self, text, size, x, y):
        font = self.pygame.font.Font("freesansbold.ttf", size)
        text_surface, text_rectangle = self.text_objects(text, font)
        text_rectangle.center = (x, y)
        self.gameDisplay.blit(text_surface, text_rectangle)

    def crash(self, x, y):
        # self.gameDisplay.blit(self.crash_img, (x, y))
        # self.message_display("You Crashed", 115, self.display_width / 2, self.display_height / 2)
        self.pygame.display.update()
        time.sleep(2)
        # self.gameloop()  # for restart the game

    def playgame(self):
        self.reset()
        observation, reward, done, _ = self.step(1)
        tot_reward = 0.0
        state_predict = observation
        while not done:
            Q = self.actor.predict(state_predict)
            action = np.argmax(Q[0])
            print(Q[0], action)
            observation_new, reward, done, info = self.step(action, ai_control=True)
            state_predict = observation_new
            tot_reward += reward
        print('Game ended! Total reward: {}'.format(tot_reward))
