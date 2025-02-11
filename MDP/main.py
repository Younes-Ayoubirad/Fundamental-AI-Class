import numpy as np
import pygame
from environment import PygameInit, AngryBirds
import matplotlib.pyplot as plt

if __name__ == "__main__":

    mean_all_reward = []

    for i in range(3):
        print(f"environment: {i}")
        env = AngryBirds()
        state = env.reset()
        policy_0 = env.value_iteration(state, gamma=0.8, theta=1e-2)

        policy_list = []
        policy_list.append(policy_0)

        env_reward = []
        for j in range(5):
            state = env.reset()
            policy = policy_list[0]

            FPS = 20
            screen, clock = PygameInit.initialization()
            running = True

            grid = env.grid
            r, c = state
            full_reward = 0

            killed_pig = 0
            step_num = 0

            while running:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                env.render(screen)

                if step_num > 50:
                    step_num = 0
                    policy = env.value_iteration(state, gamma=0.8, theta=1e-2)

                r, c = state
                action = policy[r][c]
                state, probability, reward_episode, done = env.step(action)

                step_num += 1
                if reward_episode == 250:
                    step_num = 0
                    killed_pig = killed_pig + 1
                    if j == 0:
                        policy = env.value_iteration(state, gamma=0.8, theta=1e-2)
                        policy_list.append(policy)
                    else:
                        policy = policy_list[killed_pig]

                full_reward += reward_episode
                if done:
                    env_reward.append(full_reward)
                    print(f"Episode finished with reward: {full_reward}")
                    break

                pygame.display.flip()
                clock.tick(FPS)

        mean_env_reward = np.mean(env_reward)
        print(f"Environment mean Reward:  {mean_env_reward}")

        mean_all_reward.append(mean_env_reward)

        pygame.quit()
    mean_all_env_reward = np.mean(mean_all_reward)
    print(f"All Environment mean Reward:  {mean_all_env_reward}")

    current_state = (0, 0)

    def reward_function(self):
        current_state = self.current_state
        grid = self.grid
        reward_map = [[0 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        x_start, y_start = current_state
        pig_positions = []

        for i in range(self.__grid_size):
            for j in range(self.__grid_size):
                if grid[i][j] == 'P':
                    pig_positions.append((i, j))
                elif grid[i][j] == 'Q':
                    if (i, j) != (6, 6) or (7, 6) or (6, 7):
                        reward_map[i][j] = -0.5
                    else:
                        reward_map[i][j] = -0.05
                elif grid[i][j] == 'R':
                    reward_map[i][j] = -0.01

        if pig_positions:
            min_distance = float('inf')
            nearest_pig = None
            for px, py in pig_positions:
                distance = abs(x_start - px) + abs(y_start - py)
                if distance < min_distance:
                    min_distance = distance
                    nearest_pig = (px, py)

            px, py = nearest_pig
            reward_map[px][py] = 1
        else:
            reward_map[7][7] = 1

        return reward_map

    def value_iteration(self, state, gamma=0.9, theta=1e-2):
        self.current_state = state
        self.reward_map = self.reward_function()
        self.transition_table = self.__calculate_transition_model(self.__grid_size, self.__probability_dict,
                                                                  self.reward_map)
        grid = self.grid
        n_actions = 4
        grid_size = self.__grid_size
        transition_table = self.transition_table
        reward_map = self.reward_map

        V_history = []

        prev_V = np.zeros((grid_size, grid_size))
        policy = np.zeros((grid_size, grid_size, n_actions))

        while True:
            q = np.zeros((grid_size, grid_size, n_actions))

            for r in range(grid_size):
                for c in range(grid_size):
                    state = (r, c)

                    if grid[r][c] == 'R' or grid[r][c] == 'G':
                        continue

                    for action in range(n_actions):
                        for prob, next_state, reward in transition_table[state][action]:
                            next_r, next_c = next_state
                            q[r, c, action] += (prob * (reward + (gamma * prev_V[next_r, next_c])))

            new_V = np.max(q, axis=2)

            delte = np.max(np.abs(new_V - prev_V))
            V_history.append(delte)
            if delte < theta:
                break

            prev_V = new_V.copy()

        policy = np.argmax(q, axis=2)


        # plt.imshow(new_V, cmap='viridis', interpolation='none')
        # for i in range(new_V.shape[0]):
        #     for j in range(new_V.shape[1]):
        #         plt.text(j, i, f'{new_V[i, j]:.2f}', ha='center', va='center', color='white')
        #
        # plt.colorbar(label='Value')
        # plt.title("Value Table")
        # plt.show()
        #
        #
        #
        # plt.plot(V_history)
        # plt.title("Convergence Table")
        # plt.show()
        #
        #
        # plt.imshow(policy, cmap='viridis', interpolation='none')
        # for i in range(policy.shape[0]):
        #     for j in range(policy.shape[1]):
        #         plt.text(j, i, f'{policy[i, j]}', ha='center', va='center', color='white')
        #
        # plt.colorbar(label='Value')
        # plt.title("Policy Table")
        # plt.show()
        #
        return policy