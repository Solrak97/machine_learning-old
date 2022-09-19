import random
import torch.nn as nn
from PIL import Image
from actions import Actions
import numpy as np
import torch


class Agent():
    def __init__(self):

        # Agent Settings
        self.empty_reward = -10
        self.kill_reward = 100
        self.discount_factor = 1e-3
        self.eps_greedy = 1
        self.decay = 1e-3

        random.seed(1337)

        self.action_map = [Actions.MOVE_LEFT,
                           Actions.MOVE_RIGHT, Actions.SHOOT]

        # rewind memory
        self.memory = {}

        # Agent networks
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.prediction_network = DQAgent().to(self.device)
        self.target_network = DQAgent().to(self.device)
        self.optimizer = torch.optim.Adam(self.prediction_network.parameters())


    def store_model(self, path):
        model_params = {}
        model_params['prediction_network'] = self.prediction_network.state_dict()
        model_params['target_network'] = self.target_network.state_dict()

        torch.save(model_params, path)


    def load_model(self, path):
        model_params = torch.load(path, map_location=self.device)
        self.prediction_network.load_state_dict(
            model_params['prediction_network'])
        self.target_network.load_state_dict(model_params['target_network'])

    # Transforms str into torch tensor


    def to_tensor(self, str_img):
        img = Image.frombytes('RGB', (640, 480), str_img, 'raw')
        img_array = np.array(img).reshape((3, 640, 480))
        state = torch.tensor(img_array).to(
            self.device).type(torch.float32).unsqueeze(dim=0)
        return state


    def q_val2act(self, q_val):
        idx = torch.argmax(q_val)
        return self.action_map[idx]


    def calc_loss(self, reward, q_value, next_q):
        return self.decay * (reward + self.discount_factor * torch.max(next_q) - torch.max(q_value)) ** 2


    def play(self, env):
        while (not env.is_terminal_state()):
            self.step(env, learn=False)


    def simulation(self, env):
        while (not env.is_terminal_state()):
            self.step(env, learn=True)

        if self.eps_greedy >= self.decay:
            self.eps_greedy -= self.decay


    def step(self, env, learn=True):
        state = self.to_tensor(env.get_state())
        q_val = self.prediction_network(state)

        action = self.q_val2act(q_val)

        if learn:

            if random.random() < self.eps_greedy:
                action = random.sample(self.action_map, 1)[0]

            state, reward = env.preform_action(action)

            state = self.to_tensor(state)

            next_q = self.target_network(state)

            reward *= self.kill_reward
            reward += self.empty_reward

            loss = self.calc_loss(reward, q_val, next_q)

            loss.backward()

            self.optimizer.step()

        else:
            env.preform_action(action)

    # Recibe el reward y ejecuta la corrección de error

    def update_networks(self):
        self.target_network.load_state_dict(
            self.prediction_network.state_dict())
        pass


# Red neuronal del agente
class DQAgent(nn.Module):
    def __init__(self):
        super(DQAgent, self).__init__()

        self.cl1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5)
        self.cl2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.cl3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.cl4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5)
        self.cl5 = nn.Conv2d(in_channels=96, out_channels=194, kernel_size=5)

        self.feature_activator = nn.LeakyReLU()
        self.feature_pool = nn.MaxPool2d(2, 2)

        self.classification = nn.Linear(in_features=194*16*11, out_features=3)

    def forward(self, x):
        x = self.feature_pool(self.feature_activator(self.cl1(x)))
        x = self.feature_pool(self.feature_activator(self.cl2(x)))
        x = self.feature_pool(self.feature_activator(self.cl3(x)))
        x = self.feature_pool(self.feature_activator(self.cl4(x)))
        x = self.feature_pool(self.feature_activator(self.cl5(x)))

        x = x.view(-1, 194 * 16 * 11)

        action = self.classification(x)

        return action
