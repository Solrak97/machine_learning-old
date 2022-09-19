import pygame as pg
from actions import Actions
from bullet import Bullet


class Player(pg.sprite.Sprite):

    def __init__(self, pos, image, bullet_group, scale=(32, 32), limits = (640, 480)):
        super().__init__()
        
        # Sprite
        (self.x_limit, self.y_limit) = limits
        (self.width, self.height) = scale
        self.image = pg.image.load(image).convert_alpha()
        self.image = pg.transform.scale(self.image, scale)
        self.rect = self.image.get_rect(center=pos)

        # Movement
        self.velocity = 5
        self.movement = 0
        self.max_speed = 10

        # Bullet group
        self.bullet_group = bullet_group

        # Frames
        self.shoot_frame = 0
        self.shoot_thresh = 5

    
    def move(self, velocity):
        self.movement += velocity
        self.movement = max(-self.max_speed, min(self.movement, self.max_speed))

        if (self.rect.x + self.movement) < 0:
            self.rect.x = self.width/2

        elif (self.rect.x + self.movement) > self.x_limit:
            self.rect.x = self.x_limit - self.width
        
        else:
            self.rect.x += self.movement
        
        
    def act(self, action):
        if action == Actions.MOVE_LEFT:
            self.move_left()
        if action == Actions.MOVE_RIGHT:
            self.move_right()
        if action == Actions.SHOOT:
            self.shoot()


    def move_left(self):
        self.move(-self.velocity)
        

    def move_right(self):
        self.move(self.velocity)
    
    
    def shoot(self):
        if self.shoot_frame >= self.shoot_thresh:
            self.bullet_group.add(Bullet('Assets/bala1.png', pos=(self.rect.x + self.width / 2, self.rect.y), scale=(10, 10)))
            self.shoot_frame = 0


    def update(self):
        self.shoot_frame += 1
           