import pygame
import math
import os
import cv2
import pygame.camera
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import random
# pygame.camera.init()
# cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
# cam.start()
screen_width = 1920
screen_height = 1080
clock = pygame.time.Clock()
p_height = 10
p_width = 10
camera_pos = (0, 0)
agent_dir = {
    0:'W', 1:'E', 2:'N', 3:'S', 4:'SE', 5:'NE', 6:'SW', 7:'NW', 8:'STAY' 
}
class Player:
    def __init__(self):
        pygame.init()
        self.x_cor = 800
        self.y_cor = 500
        self.speed = 40/6
        self.x = 800
        self.y = 500
    
    def move(self, keys, display, mode):
        if mode == 'human':
            keys=pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.x_cor -= self.speed
            if keys[pygame.K_RIGHT]:
                self.x_cor +=self.speed
            if keys[pygame.K_UP]:
                self.y_cor -=self.speed
            if keys[pygame.K_DOWN]:
                self.y_cor +=self.speed
            if(self.x_cor < -11000):
                self.x_cor = -11000
            if self.y_cor < -11000:
                self.y_cor = -11000
            if self.x_cor > 10000:
                self.x_cor = 10000
            if self.y_cor > 10000:
                self.y_cor = 10000
        else:
            if keys == 0:                     #West
                self.x_cor -= self.speed
            if keys == 1:                     #East
                self.x_cor +=self.speed
            if keys == 2:                     #North
                self.y_cor -=self.speed
            if keys == 3:                     #South
                self.y_cor +=self.speed
            if keys == 4:                     #South-East
                self.x_cor += self.speed
                self.y_cor += self.speed
            if keys == 5:                     #North-East
                self.x_cor += self.speed
                self.y_cor -= self.speed
            if keys == 6:                     #South-West
                self.x_cor -= self.speed
                self.y_cor += self.speed
            if keys == 7:                     #North-West
                self.x_cor -= self.speed
                self.y_cor -= self.speed
            if(self.x_cor < -11000):
                self.x_cor = -11000
            if self.y_cor < -11000:
                self.y_cor = -11000
            if self.x_cor > 10000:
                self.x_cor = 10000
            if self.y_cor > 10000:
                self.y_cor = 10000
    def render(self, display):
        pygame.draw.circle(display, (255,0,0), (self.x,self.y),10)


class Garden:
    def __init__(self):
        pygame.init()
        self.counter = 300
        pygame.time.set_timer(pygame.USEREVENT, 1000)
        self.font = pygame.font.SysFont('Consolas', 30)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        # self.screen.fill((0, 255, 0))
        self.clock = pygame.time.Clock()
        self.game_speed = 60
        self.player = Player()
        self.done = False
        self.curr_action = 0

    def fill_array(self, vec, angle, dist, q):
        arr = np.zeros(8)
        for i in range(8):
            arr[i] = np.sqrt(20000**2+20000*2)
        if vec[0] > 0 and vec[1] > 0:
            arr[5] = min(arr[5], dist)
            if angle < 45:
                arr[1] = min(arr[1], dist)
            else:
                arr[2] = min(arr[2], dist)
        if vec[0] > 0 and vec[1] < 0:
            arr[4] = min(arr[4], dist)
            if angle < 45:
                arr[1] = min(arr[1], dist)
            else:
                arr[3] = min(arr[3], dist)
        if vec[0] < 0 and vec[1] > 0:
            arr[7] = min(arr[7], dist)
            if angle < 45:
                arr[0] = min(arr[0], dist)
            else:
                arr[2] = min(arr[2], dist)
        if vec[0] < 0 and vec[1] < 0:
            arr[6] = min(arr[6], dist)
            if angle < 45:
                arr[0] = min(arr[0], dist)
            else:
                arr[3] = min(arr[3], dist)
        
        if q == 4:
            q = 8
        elif q == 3:
            q = 5
        elif q == 2:
            q = 2
        else:
            q = 0.5
        m = np.min(arr)
        for i in range(8):
            if arr[i] == m:
                arr[i] = ((5/m)*q) + np.random.normal(0,0.5)
            else: arr[i] = 0

        for i in range(8):
            if arr[i] > 15000:
                arr[i] = 0
        return arr


    def drawBerries(self, berx_cor, bery_cor, sz):
        sz_4 = np.zeros(8)
        sz_3 = np.zeros(8)
        sz_2 = np.zeros(8)
        sz_1 = np.zeros(8)
        density = 0
        min_dist = np.zeros(8)
        for i in range(4):
            min_dist[i] = 200000

        for i in range(len(berx_cor)):
            curr_x = (berx_cor[i]+self.player.x) - (self.player.x_cor)
            curr_y = (bery_cor[i]+self.player.y) - (self.player.y_cor)
            # print(curr_x, curr_y)
            if(curr_x >= 0 and curr_x <= screen_width and curr_y >= 0 and curr_y <= screen_height):
                density += sz[i]*sz[i]*np.pi*2
                dist = np.sqrt((curr_x - self.player.x)**2 + (curr_y - self.player.y)**2)*0.01
                vec = [curr_x - self.player.x, -curr_y + self.player.y]
                if vec[0] != 0:
                    angle = np.rad2deg(np.arctan(abs(vec[1])/abs(vec[0])))
                else: angle = 90
                if sz[i] == 40:
                    if dist < min_dist[3]:
                        sz_4 = self.fill_array(vec, angle, dist, 4)
                        min_dist[3] = dist
                if sz[i] == 30:
                    if dist < min_dist[2]:
                        sz_3 = self.fill_array(vec, angle, dist, 3)
                        min_dist[2] = dist
                if sz[i] == 20:
                    if dist < min_dist[1]:
                        sz_2 = self.fill_array(vec, angle, dist, 2)
                        min_dist[1] = dist
                if sz[i] == 10:
                    if dist < min_dist[0]:
                        sz_1 = self.fill_array(vec, angle, dist, 1)
                        min_dist[0] = dist
                pygame.draw.circle(self.screen, (0, 0, 255), (curr_x, curr_y), sz[i])
        
        return sz_4, sz_3, sz_2, sz_1 , density/(1920*1080)

    def action(self, action, mode):
        self.player.move(action, self.screen, mode)
        self.curr_action = action
        return self.player.x_cor, self.player.y_cor, self.done, self.counter

    def view(self, berx_cor, bery_cor, sz, juice):
        font = pygame.font.SysFont("comicsans", 30, True)
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT: 
                self.counter -= 1
            if event.type == pygame.QUIT or self.counter == 0 or juice < 0:
                self.done = True
                # pygame.quit()
                
        self.screen.fill((0, 255, 0))
        sz_4, sz_3, sz_2, sz_1, density = self.drawBerries(berx_cor, bery_cor, sz)
        self.player.render(self.screen)
        temp_str = str(juice)
        temp_str = temp_str[0:4]
        temp_str = temp_str + " " + str(self.counter)
        text = font.render("Juice, Time: " + temp_str, 1, (0,0,0)) # Arguments are: text, anti-aliasing, color
        text_rect = text.get_rect(center = self.screen.get_rect().center)
        self.screen.blit(text, (text_rect[0]+800, text_rect[1]-500))
        text = font.render("Action: " + agent_dir[self.curr_action], 1, (0,0,0)) # Arguments are: text, anti-aliasing, color
        text_rect = text.get_rect(center = self.screen.get_rect().center)
        self.screen.blit(text, (text_rect[0]-900, text_rect[1]-500))
        pygame.display.flip()
        # pygame.image.save(self.screen, "screenshot.png")
        self.clock.tick(60)
        # img_array = np.array(pygame.surfarray.array2d(self.screen))
        # img_array = img_array/np.max(img_array)
        # img_array = img_array.swapaxes(0,1)
        # img_array = img_array*255
        # img_array = np.array(img_array, dtype = np.uint8)
        # final_img = cv2.resize(img_array, (100, 100),
        #         interpolation = cv2.INTER_NEAREST)
        # cv2.imwrite('temp_img.png', final_img)
        img_array = 0
        # print(img_array)
        next_state = []
        for i in sz_4:
            next_state.append(i)
        for i in sz_3:
            next_state.append(i)
        for i in sz_2:
            next_state.append(i)
        for i in sz_1:
            next_state.append(i) 
        next_state.append(density*100)   
        next_state = np.array(next_state)
        return img_array, next_state, 300 - self.counter
        # pxarray = pygame.PixelArray(self.screen)
        # img = cam.get_image()
        # img_array = pygame.surfarray.pixels3d(pxarray)
        # cv2.imwrite('screen.png', pxarray)
