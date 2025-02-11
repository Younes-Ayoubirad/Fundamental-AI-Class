import csv
from game import AngryGame
from minimax_agent import MinimaxAgent


def extract_features(grid, agent):
    features = {
        "distance_hen": agent.find_closet_path_hen(grid),
        "distance_egg": agent.find_closet_path_egg(grid),
        # "distance_shooter": agent.find_closet_path_shooter(grid),
        "num_eggs": len(AngryGame.get_egg_coordinate(grid)),
        "num_pigs": len(AngryGame.get_pig_coordinate(grid)),
    }
    return features


if __name__ == "__main__":
    env = AngryGame(template='hard')
    agent = MinimaxAgent(env, max_depth=3)

    csv_file_name = "training_data.csv"

    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # writer.writerow(["distance_hen", "distance_egg", "distance_shooter", "num_eggs", "num_pigs", "score"])
        writer.writerow(["distance_hen", "distance_egg", "num_eggs", "num_pigs", "score"])

        num_games = 100

        for i in range(num_games):
            print(f"game : {i}")
            env.reset()

            counter = 0
            while not (AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions)):
                if counter % 2 == 0:
                    grid = env.grid
                    score = agent.evaluate(grid)
                    features = extract_features(grid, agent)
                    # writer.writerow([features["distance_hen"], features["distance_egg"],
                    #                  features["distance_shooter"], features["num_eggs"],
                    #                  features["num_pigs"], score])

                    action, score2 = agent.get_best_action(env.grid)

                    writer.writerow([features["distance_hen"], features["distance_egg"],
                                     features["num_eggs"], features["num_pigs"], score])
                    # writer.writerow([features["distance_hen"], features["distance_egg"],
                    #                  features["distance_shooter"], features["num_eggs"],
                    #                  features["num_pigs"], score3])
                    env.hen_step(action)
                    if AngryGame.is_win(env.grid):
                        break

                if counter % 2 == 1:
                    env.queen_step()
                    if AngryGame.is_lose(env.grid, env.num_actions):
                        break

                counter += 1

