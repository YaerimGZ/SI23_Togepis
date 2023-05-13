import gym
import numpy as np
import time

class Agent():
    def __init__(self, env, params=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print("Posibles acciones del agente: ", self.action_space)

        # Tabla de observaciones x acciones
        self.q_table = np.zeros((self.observation_space.n,
                                 self.action_space.n))
        
        # Parámetros
        self.epsilon= 0.5  # Probabilidad de exploración
        self.alpha = 0.8  # Learning rate
        self.gamma = 0.9  # Descuento

    def step(self, obs):
        '''
            Toma una acción dada la observación
        '''
        # Valor aleatorio
        random_value = np.random.uniform(0, 1)
        if random_value < self.epsilon:
            # random
            action = self.env.action_space.sample()
        else:
            # greedy
            q_values = self.q_table[obs]
            action = np.argmax(np.random.rand(q_values.shape[0]) * (q_values == q_values.max()))
        return action

    def episode_rollout(self, max_steps):
        episode_return = 0
        curr_obs = self.env.reset()
        for i in range(max_steps):
            self.env.render()

            # tomar acción
            action = self.step(curr_obs)

            # Obtener estado siguiente
            next_obs, reward, done, info = self.env.step(action)

            # Actualizar la tabla q
            self.q_table[curr_obs, action] = self.q_table[curr_obs, action] + \
                self.alpha * (reward + \
                              self.gamma * np.max(self.q_table[next_obs]) -\
                              self.q_table[curr_obs, action])

            episode_return += reward
            obs = next_obs
            if done:
                break
            time.sleep(0.5)
        print("Tabla Q: ", self.q_table)
        return episode_return, i