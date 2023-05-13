import gym
import numpy as np
import time

class Agent():
    def __init__(self, env, params=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print("Posibles acciones del agente: ", self.action_space)

    def step(self, obs):
        '''
            Toma una acción dada la observación
        '''
        # TODO: Regresa un valor aleatorio valido
        # Despues implementa una politica apropiada para q learning

        return

    def episode_rollout(self, max_steps):
        episode_return = 0
        curr_obs = self.env.reset()
        # TODO: Realiza un rollout de un episodio
        # Rgresar el retorno y la cantidad de timesteps
        # Despues implementa q-learning

        return ()