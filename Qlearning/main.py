import math

import numpy as np
import pygame
from environment import UnknownAngryBirds, PygameInit
import matplotlib.pyplot as plt


def get_index(pig):
    number = int(''.join('1' if x else '0' for x in pig), 2)
    return number


def get_state(state):
    return state[1]*8+state[0]


if __name__ == "__main__":
    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    pig = 256
    num_states = 64
    num_actions = 4
    q_table = np.zeros((pig, num_states, num_actions), dtype=np.int64)
    learning_rate = 0.05
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9997
    epsilon_min = 0.1
    episode_reward = []
    env.render(screen)
    for episode in range(10000):
        pigs = [True, True, True, True, True, True, True, True]
        state = env.reset()
        running = True
        total_reward = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            previous_pig = get_index(pigs)
            previous_state_index = get_state(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(q_table[previous_pig,previous_state_index, :])

            next_state, reward, pig_state, done = env.step(action)
            next_pig = get_index(pig_state)
            next_state_index = get_state(next_state)
            q_table[previous_pig, previous_state_index, action] = (q_table[previous_pig, previous_state_index, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_pig, next_state_index, :]) -q_table[previous_pig, previous_state_index, action]))
            pigs = pig_state
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode + 1} finished with reward: {total_reward}")
                if episode == 9995:
                    epsilon = 0
                    epsilon_min = 0
                episode_reward.append(total_reward)
                running = False


        pygame.display.flip()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    plt.scatter(range(len(episode_reward)), episode_reward, color='b', marker='o')
    plt.title("reward table")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.grid(True)
    plt.show()
    input()


