import pygame as pg
from actions import Actions
from game import Game
from agent import Agent

# Setup
SCREEN_SIZE = (640, 480)
FPS = 60

pg.init()

clock = pg.time.Clock()
icon = pg.image.load('Assets/green.png')
pg.display.set_caption('Space Invaders')
pg.display.set_icon(icon)

screen = pg.display.set_mode(SCREEN_SIZE)
pg.mixer.music.load("Assets/music.wav")
pg.mixer.music.play()


# Creación del modelo
agent = Agent()   

game_sims = 1000

for sim in range(game_sims):

    
    game = Game(screen, SCREEN_SIZE)
    agent.simulation(game)
    print(f"########################ITERATION N {sim} ###################################")
    print(f"Score {game.score}")

    if sim % 10 == 0:
        agent.update_networks()
        agent.store_model('model3.pth')
