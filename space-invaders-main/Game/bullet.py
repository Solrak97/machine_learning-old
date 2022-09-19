import pygame as pg

class Bullet(pg.sprite.Sprite):

    def __init__(self, image, pos, scale = (32, 32), limits = (640, 480)):
        super().__init__()
        # Sprite
        (self.x_limit, self.y_limit) = limits
        (self.width, self.height) = scale
        self.image = pg.image.load(image).convert_alpha()
        self.image = pg.transform.scale(self.image, scale)
        self.rect = self.image.get_rect(center=pos)
        pass

    def update(self):
        self.rect.y -= 7

        if self.rect.y < 0:
            self.kill()
        
    pass