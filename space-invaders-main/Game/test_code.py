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
agent.load_model('model3.pth')


# Se corren 10 simulaciones del juego
game_sims = 10

for sim in range(game_sims):

    game = Game(screen, SCREEN_SIZE)
    agent.play(game)