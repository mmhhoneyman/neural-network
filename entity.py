import os
from constants import SimConfig, EntityConfig, NNConfig
import pygame
from neural_network import NeuralNetwork

class Entity:
    collision_trimming = EntityConfig.COLLISION_TRIMMING
    wall_trimming = EntityConfig.WALL_TRIMMING

    def __init__(self, x_pos, y_pos, color, neural_network = None, width = 50, height = 50):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.color = color
        self.neural_network = neural_network
        self.width = width
        self.height = height
        self.move_x = 0
        self.move_y = 0

    def move(self):
        warp = False
        self.x_pos += int(self.move_x)
        self.y_pos += int(self.move_y)
        if(SimConfig.INPUTS_ALLOWED):
            self.move_x = 0
            self.move_y = 0

        if(self.x_pos + self.width - self.wall_trimming > SimConfig.SCREEN_WIDTH):
            self.x_pos = -self.wall_trimming
            warp = True

        if(self.x_pos < -self.wall_trimming):
            self.x_pos = SimConfig.SCREEN_WIDTH - self.width + self.wall_trimming
            warp = True
        
        if(self.y_pos + self.height - self.wall_trimming > SimConfig.SCREEN_HEIGHT):
            self.y_pos = -self.wall_trimming
            warp = True

        if(self.y_pos < -self.wall_trimming):
            self.y_pos = SimConfig.SCREEN_HEIGHT - self.height + self.wall_trimming
            warp = True

        return warp

    def check_collision(self, entity):
        if (
            self.x_pos + self.collision_trimming < entity.x_pos + entity.width and
            self.x_pos + self.width - self.collision_trimming > entity.x_pos and
            self.y_pos + self.collision_trimming < entity.y_pos + entity.height and
            self.y_pos + self.height - self.collision_trimming > entity.y_pos
        ):
            return True
        else:
            return False
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x_pos, self.y_pos, self.width, self.height))
        