# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import enum
import numpy as np
from PIL import Image, ImageTk
import random
import time
try:
    import tkinter as tk
except ImportError:
    try:
        import Tkinter as tk
    except ImportError:
        print("Unsupported library: Tkinter, please install")

from tkinter import simpledialog

# CUSTOMIZABLE PARAMETERS

# Maze related
OUT_OF_BOUNDS_REWARD = -1000
EMPTY_REWARD = -1
KEY_REWARD = 100
GOAL_REWARD = 1000
MINSIZE = 8
MAXSIZE = 15

# UI related
FPS = 24
APS = 2

colors = {
    0: (32, 32, 32),  # Wall
    1: (220, 220, 220),  # Path
    2: (255, 0, 0),  # Agent
    3: (98, 208, 255),  # Entry
    4: (0, 162, 233),  # Exit
    5: (222, 222, 0),  # Key
}


class Agent():
    # Initializes the agent
    def __init__(self, seed, state_dims, actions, learning_rate, discount_factor, eps_greedy, decay):
        # Use self.prng any time you require to call a random funcion
        self.prng = random.Random()
        self.prng.seed(seed)

        self.state_dims = state_dims
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps_greedy = eps_greedy
        self.decay = decay

        qtable_m = 1
        for e in [*state_dims]:
            qtable_m *= e

        self.qtable = np.zeros((qtable_m, len(actions)))

    def state_to_index(self, state):

        if len(state) == 2:
            x, y = state
            w = self.state_dims[0]
            return y * w + x

        else:
            x, y, z = state
            x_max = self.state_dims[0]
            y_max = self.state_dims[1]
            return (z * x_max * y_max) + (y * x_max) + x;

    def select_action(self, q_index):
        epsilon = self.prng.random()
        if epsilon < self.eps_greedy:
            action_index = self.prng.randint(0, 3)

        else:
            action_index = np.argmax(self.qtable[q_index])
        return action_index

    def simulation(self, env):

        # Simular una ejecución
        while (not env.is_terminal_state()):
            self.step(env, learn=True)

        if self.eps_greedy >= self.decay:
            self.eps_greedy -= self.decay

    # Performs a single step of the simulation by the agent, if learn=False no updates are performed

    def step(self, env, learn=True):
        state = env.get_state()
        q_index = self.state_to_index(state)

        if learn:
            action_index = self.select_action(q_index)
            reward, state = env.perform_action(self.actions[action_index])
            

            # Update
            q_index_prime = self.state_to_index(state)
            self.qtable[q_index][action_index] = (self.qtable[q_index][action_index] + self.learning_rate * (
                reward + self.discount_factor * np.max(self.qtable[q_index_prime]) - self.qtable[q_index][action_index]))

        else:
            action_index = np.argmax(self.qtable[q_index])
            env.perform_action(self.actions[action_index])


### DO NOT MODIFY ANYTHING ELSE ###

class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Maze():
    def __init__(self, map_seed, addKey=False):
        self.rnd = random.Random()
        self.rnd.seed(map_seed)
        self.w = self.rnd.randint(MINSIZE, MAXSIZE)
        self.h = self.rnd.randint(MINSIZE, MAXSIZE)
        self.height, self.width = self.h+2, self.w+2
        self.board = np.zeros((self.height, self.width))
        flip = self.rnd.randint(0, 1)
        if self.rnd.randint(0, 1):
            self.entry = (1+int(self.rnd.random()*self.h),
                          flip*(self.w+1) + (-1 if flip else 1))
        else:
            self.entry = (flip*(self.h+1) + (-1 if flip else 1),
                          1+int(self.rnd.random()*self.w))
        walls = [self.entry]
        valid = []
        while walls:
            wall = walls.pop(int(self.rnd.random()*len(walls)))
            if 2 > ((self.board[(wall[0]-1, wall[1])] > 0)*1 + (self.board[(wall[0]+1, wall[1])] > 0)*1 + (self.board[(wall[0], wall[1]-1)] > 0)*1 + (self.board[(wall[0], wall[1]+1)] > 0)*1):
                self.board[wall] = 1
                valid.append(wall)
            else:
                continue
            if wall[0]-1 > 0:
                walls.append((wall[0]-1, wall[1]))
            if wall[0]+1 <= self.h:
                walls.append((wall[0]+1, wall[1]))
            if wall[1]-1 > 0:
                walls.append((wall[0], wall[1]-1))
            if wall[1]+1 <= self.w:
                walls.append((wall[0], wall[1]+1))
        self.board[self.entry] = 3
        ext = self.entry
        while ext == self.entry:
            ext = self.rnd.choice(valid)
        self.board[ext] = 4
        self.position = self.entry
        self.addKey = addKey
        self.hasKey = 0 if addKey else 1
        if addKey:
            key = ext
            while self.board[key] != 1:
                key = self.rnd.choice(valid)
            self.board[key] = 5

    def get_board(self, showAgent=True):
        res = self.board.copy()
        if showAgent:
            res[self.position] = 2
        return res

    # Resets the environment
    def reset(self):
        self.position = self.entry
        self.hasKey = 0 if self.addKey else 1

    # Returns state-space dimensions
    def get_state_dimensions(self):
        if self.addKey:
            return (self.height, self.width, 2)
        else:
            return (self.height, self.width)

    # Returns action list
    def get_actions(self):
        return [a.value for a in Action]

    # Return current state as a tuple
    def get_state(self):
        if self.addKey:
            return (*self.position, self.hasKey)
        else:
            return self.position

    # Returns whether current state is terminal
    def is_terminal_state(self):
        return self.board[self.position] == 0 or (self.board[self.position] == 4 and self.hasKey)

    # Performs an action and returns its reward and the new state
    def perform_action(self, action):
        if action == Action.UP:
            self.position = (self.position[0]-1, self.position[1])
        elif action == Action.DOWN:
            self.position = (self.position[0]+1, self.position[1])
        elif action == Action.RIGHT:
            self.position = (self.position[0], self.position[1]+1)
        elif action == Action.LEFT:
            self.position = (self.position[0], self.position[1]-1)
        space = self.board[self.position]
        if space == 0:
            return OUT_OF_BOUNDS_REWARD, self.get_state()
        elif space == 4 and self.hasKey:
            return GOAL_REWARD, self.get_state()
        elif space == 5 and self.hasKey == 0:
            self.hasKey = 1
            return KEY_REWARD, self.get_state()
        return EMPTY_REWARD, self.get_state()


class mainWindow():
    def __init__(self, agentClass):
        self.map_seed = random.randint(0, 65535)
        self.maze = Maze(self.map_seed)
        self.agent_seed = random.randint(0, 256)
        self.agentClass = agentClass
        # Control
        self.redraw = False
        self.playing = False
        self.simulations = 0
        self.learning_rate = 0.01
        self.discount = 0.5
        self.greedy = 0.8
        self.decay = 1e-7
        self.agent = self.agentClass(self.agent_seed, self.maze.get_state_dimensions(
        ), self.maze.get_actions(), self.learning_rate, self.discount, self.greedy, self.decay)
        # Interface
        self.root = tk.Tk()
        self.root.title("Maze AI")
        self.root.bind("<Configure>", self.resizing_event)
        self.frame = tk.Frame(self.root, width=700, height=550)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=1, height=1)
        # Simulation control
        self.labelControl = tk.Label(
            self.frame, text="Control", relief=tk.RIDGE, padx=5, pady=2)
        self.stringSimulations = tk.StringVar(
            value="Simulations: "+str(self.simulations))
        self.labelSimulations = tk.Label(
            self.frame, textvariable=self.stringSimulations, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonReset = tk.Button(
            self.frame, text="Reset", command=self.reset, bg="sea green")
        self.buttonNext = tk.Button(
            self.frame, text="Next", command=self.buttonNext_press, bg="sea green")
        self.buttonSkip = tk.Button(
            self.frame, text="Skip", command=self.buttonSkip_press, bg="sea green")
        self.buttonRun = tk.Button(
            self.frame, text="Run", command=self.buttonRun_press, bg="forest green")
        # Seeds label
        self.labelSeeds = tk.Label(
            self.frame, text="Seeds", relief=tk.RIDGE, padx=5, pady=2)
        # Agent seed: agent seed button, agent seed string and label
        self.stringAgentseed = tk.StringVar(
            value="Agent seed: "+str(self.agent_seed))
        self.labelAgentseed = tk.Label(
            self.frame, textvariable=self.stringAgentseed, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonSetAgentseed = tk.Button(
            self.frame, text="Set", command=self.buttonSetAgentseed_press, bg="sea green")
        # Map seed: set map seed button, new map seed button, map seed string and label
        self.buttonSetMapseed = tk.Button(
            self.frame, text="Seed", command=self.buttonSetMapseed_press, bg="indian red")
        self.buttonNewMapseed = tk.Button(
            self.frame, text="Random", command=self.buttonNewMapseed_press, bg="indian red")
        self.stringMapseed = tk.StringVar(
            value="Map seed: "+str(self.map_seed))
        self.labelMapseed = tk.Label(
            self.frame, textvariable=self.stringMapseed, relief=tk.RIDGE, padx=5, pady=2)
        # Customization
        self.labelCustomization = tk.Label(
            self.frame, text="Customization", relief=tk.RIDGE, padx=5, pady=2)
        # Alpha learning rate, Gamma discount, Epsilon greedy
        self.stringAlphalr = tk.StringVar(
            value="α-learning: "+str(self.learning_rate))
        self.labelAlphalr = tk.Label(
            self.frame, textvariable=self.stringAlphalr, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonAlphalr = tk.Button(
            self.frame, text="Set", command=self.buttonAlphalr_press, bg="sea green")
        self.stringGammadisc = tk.StringVar(
            value="γ-discount: "+str(self.discount))
        self.labelGammadisc = tk.Label(
            self.frame, textvariable=self.stringGammadisc, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonGammadisc = tk.Button(
            self.frame, text="Set", command=self.buttonGammadisc_press, bg="sea green")
        self.stringEpsilongreedy = tk.StringVar(
            value="ε-greedy: "+str(self.greedy))
        self.labelEpsilongreedy = tk.Label(
            self.frame, textvariable=self.stringEpsilongreedy, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEpsilongreedy = tk.Button(
            self.frame, text="Set", command=self.buttonEpsilongreedy_press, bg="sea green")
        self.stringEtadecay = tk.StringVar(value="η-decay: "+str(self.decay))
        self.labelEtadecay = tk.Label(
            self.frame, textvariable=self.stringEtadecay, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEtadecay = tk.Button(
            self.frame, text="Set", command=self.buttonEtadecay_press, bg="sea green")
        self.keyOn = tk.IntVar()
        self.checkboxKey = tk.Checkbutton(
            self.frame, text="Add requirement (key)", variable=self.keyOn, command=self.reset, relief=tk.RIDGE)
        # Others
        # self.heatmapOn = tk.IntVar()
        # self.checkboxHeatmap = tk.Checkbutton(self.frame, text="Display heatmap", variable=self.heatmapOn, command=self.redraw_canvas, relief=tk.RIDGE)
        # Start
        self.root.after(0, self.update_loop)
        self.root.mainloop()

    # Resizing event
    def resizing_event(self, event):
        if event.widget == self.root:
            self.redraw = True
            self.canvas_width = max(event.width - 250, 1)
            self.canvas_height = max(event.height - 40, 1)
            self.frame.configure(width=event.width, height=event.height)
            self.canvas.configure(width=self.canvas_width,
                                  height=self.canvas_height)
            self.canvas.place(x=20, y=20)
            # Control
            self.labelControl.place(x=event.width - 210, y=20, width=190)
            self.labelSimulations.place(x=event.width - 210, y=50)
            self.buttonReset.place(x=event.width - 190, y=80, width=50)
            self.buttonNext.place(x=event.width - 130, y=80, width=50)
            self.buttonSkip.place(x=event.width - 70, y=80, width=50)
            self.buttonRun.place(x=event.width - 170, y=115, width=120)
            # Seeds
            self.labelSeeds.place(x=event.width - 210, y=150, width=190)
            # Agent seed
            self.labelAgentseed.place(x=event.width - 210, y=180)
            self.buttonSetAgentseed.place(x=event.width - 70, y=180)
            # Map seed
            self.labelMapseed.place(x=event.width - 210, y=215)
            self.buttonSetMapseed.place(x=event.width-180, y=250, width=60)
            self.buttonNewMapseed.place(x=event.width-100, y=250, width=60)
            # Customization
            self.labelCustomization.place(
                x=event.width - 210, y=290, width=190)
            self.labelAlphalr.place(x=event.width - 210, y=320)
            self.buttonAlphalr.place(x=event.width - 70, y=320)
            self.labelGammadisc.place(x=event.width - 210, y=350)
            self.buttonGammadisc.place(x=event.width - 70, y=350)
            self.labelEpsilongreedy.place(x=event.width - 210, y=380)
            self.buttonEpsilongreedy.place(x=event.width - 70, y=380)
            self.labelEtadecay.place(x=event.width - 210, y=410)
            self.buttonEtadecay.place(x=event.width - 70, y=410)
            self.checkboxKey.place(x=event.width - 210, y=440)
            # Others
            # self.checkboxHeatmap.place(x=event.width - 210, y=max(event.height - 50,470))

    # Update loop
    def update_loop(self):
        if self.playing:
            if (time.time()-self.last_action) >= 1/APS:
                self.last_action = time.time()
                if not self.maze.is_terminal_state():
                    self.agent.step(self.maze, learn=False)
                else:
                    self.showPlayer = not self.showPlayer
                self.redraw = True
        if self.redraw:
            self.redraw_canvas()
        self.root.after(int(1000/FPS), self.update_loop)

    # Set agent seed button
    def buttonSetAgentseed_press(self):
        if self.playing:
            return
        x = simpledialog.askinteger(
            "Agent seed", "Input agent seed:", parent=self.root, minvalue=0)
        if x and x != self.agent_seed:
            self.agent_seed = x
            self.stringAgentseed.set("Agent seed: "+str(self.agent_seed))
            self.reset()

    # Set map seed button
    def buttonSetMapseed_press(self):
        if self.playing:
            return
        x = simpledialog.askinteger(
            "Map seed", "Input map seed:", parent=self.root, minvalue=0)
        if x and x != self.map_seed:
            self.map_seed = x
            self.stringMapseed.set("Map seed: "+str(self.map_seed))
            self.reset()

    # New map seed button
    def buttonNewMapseed_press(self):
        if self.playing:
            return
        self.map_seed = random.randint(0, 65535)
        self.stringMapseed.set("Map seed: "+str(self.map_seed))
        self.reset()

    # Next button
    def buttonNext_press(self):
        if self.playing:
            return
        self.run_quick_simulation(1)

    # Skip button
    def buttonSkip_press(self):
        if self.playing:
            return
        x = simpledialog.askinteger(
            "Run simulations", "How many simulations:", parent=self.root, minvalue=1, initialvalue=10)
        if x:
            self.run_quick_simulation(x)

    # Run button
    def buttonRun_press(self):
        self.showPlayer = True
        self.buttonRun.config(text=("Run" if self.playing else "Stop"), bg=(
            "forest green" if self.playing else "orange red"))
        self.last_action = time.time()
        if not self.playing:
            self.maze.reset()
            self.redraw = True
        self.playing = not self.playing

    # Alpha-lr button
    def buttonAlphalr_press(self):
        if self.playing:
            return
        x = simpledialog.askfloat(
            "α Learning rate", "Input the learning rate:", parent=self.root, minvalue=0, maxvalue=1)
        if x:
            self.learning_rate = x
            self.stringAlphalr.set("α-learning: "+str(self.learning_rate))
            self.reset()

    # Gamma-disc button
    def buttonGammadisc_press(self):
        if self.playing:
            return
        x = simpledialog.askfloat(
            "γ Discount factor", "Input the discount factor:", parent=self.root, minvalue=0, maxvalue=1)
        if x:
            self.discount = x
            self.stringGammadisc.set("γ-discount: "+str(self.discount))
            self.reset()

    # Epsilon-greedy button
    def buttonEpsilongreedy_press(self):
        if self.playing:
            return
        x = simpledialog.askfloat(
            "ε Greedy", "Input the initial ε greedy value:", parent=self.root, minvalue=0, maxvalue=1)
        if x:
            self.greedy = x
            self.stringEpsilongreedy.set("ε-greedy: "+str(self.greedy))
            self.reset()

    # Eta-decay button
    def buttonEtadecay_press(self):
        if self.playing:
            return
        x = simpledialog.askfloat(
            "η Decay factor", "Input the η decay factor for ε:", parent=self.root, minvalue=0, maxvalue=1)
        if x:
            self.decay = x
            self.stringEtadecay.set("η-decay: "+str(self.decay))
            self.reset()

    def reset(self):
        if self.playing:
            self.buttonRun_press()
        self.maze = Maze(self.map_seed, self.keyOn.get() == 1)
        self.agent = self.agentClass(self.agent_seed, self.maze.get_state_dimensions(
        ), self.maze.get_actions(), self.learning_rate, self.discount, self.greedy, self.decay)
        self.simulations = 0
        self.stringSimulations.set("Simulations: "+str(self.simulations))
        self.redraw = True

    def run_quick_simulation(self, n):
        for i in range(n):
            self.agent.simulation(self.maze)
            self.maze.reset()
        self.simulations += n
        self.stringSimulations.set("Simulations: "+str(self.simulations))
        self.redraw = True

    def redraw_canvas(self):
        if (self.maze.width/self.maze.height)*self.canvas_height > self.canvas_width:
            self.board_width, self.board_height = self.canvas_width, int(
                (self.maze.height/self.maze.width)*self.canvas_width)
        else:
            self.board_height, self.board_width = self.canvas_height, int(
                (self.maze.width/self.maze.height)*self.canvas_height)
        self.board_offset_x, self.board_offset_y = (
            self.canvas_width - self.board_width)//2, (self.canvas_height - self.board_height)//2
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, self.canvas_width, self.canvas_height, fill="#606060", width=0)
        pixels = np.array([[colors[y] for y in x] for x in self.maze.get_board(
            showAgent=not self.playing or self.showPlayer)])
        self.image = Image.fromarray(pixels.astype('uint8'), 'RGB')
        self.photo = ImageTk.PhotoImage(image=self.image.resize(
            (self.board_width, self.board_height), resample=Image.NEAREST))
        self.canvas.create_image(
            self.board_offset_x, self.board_offset_y, image=self.photo, anchor=tk.NW)
        dy = self.board_height / self.maze.height
        dx = self.board_width / self.maze.width
        for i in range(1, self.maze.height):
            self.canvas.create_line(self.board_offset_x, self.board_offset_y+int(
                dy*i), self.board_offset_x+self.board_width, self.board_offset_y+int(dy*i))
        for i in range(1, self.maze.width):
            self.canvas.create_line(self.board_offset_x + int(dx*i), self.board_offset_y,
                                    self.board_offset_x+int(dx*i), self.board_offset_y+self.board_height)
        self.canvas.create_rectangle(self.board_offset_x, self.board_offset_y, self.board_offset_x +
                                     self.board_width, self.board_offset_y+self.board_height, outline="#0000FF", width=3)
        self.redraw = False


if __name__ == "__main__":
    x = mainWindow(Agent)
