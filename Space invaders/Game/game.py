import sys
from enemy_manager import EnemyManager
from player import Player
import pygame as pg


class Game():
    def __init__(self, display, display_size):

        # Game settings
        self.display = display
        self.display_size = display_size

        self.player_group = pg.sprite.Group()
        self.enemy_group = pg.sprite.Group()
        self.bullet_group = pg.sprite.Group()

        # Enemy manager
        self.enemy_manager = EnemyManager(self.enemy_group, self.bullet_group)

        # Player setup
        self.player = Player((self.display_size[0] / 2, 460), 'Assets/jugador.png', self.bullet_group)
        self.player_group.add(self.player)

        # Score
        self.score = 0

        # Screen rect
        self.rect = pg.Rect(0, 0, self.display_size[0], self.display_size[1])

        

    # Graphics update
    def draw_state(self):
        self.enemy_manager.draw(self.display)
        self.player_group.draw(self.display)
        self.bullet_group.draw(self.display)



    # Internal game object state update
    def update_state(self):
        self.enemy_manager.update()
        self.player_group.update()
        self.bullet_group.update()

    

    def preform_action(self, action):

        self.player.act(action)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        
        self.display.fill((0, 0, 0))
        self.draw_state()
        self.update_state()

        downs = self.enemy_manager.check_downs()
        self.score += downs
        pil_string_image = pg.image.tostring(self.display, "RGBA",False)
        pg.display.flip()
        
        return(pil_string_image, downs)      
            


    def get_state(self):
        return pg.image.tostring(self.display, "RGBA",False) 



    def is_terminal_state(self):
        return self.enemy_manager.enemy_trespasses()