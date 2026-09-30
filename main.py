import os

import copy
from constants import GameConfig, SimConfig, EntityConfig, NNConfig
import pygame
import pandas as pd
import math
import random
import sys
import json
import csv
from entity import Entity
from neural_network import NeuralNetwork

def check_inputs():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        chaser.move_y = -GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_s]:
        chaser.move_y = GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_a]:
        chaser.move_x = -GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_d]:
        chaser.move_x = GameConfig.INPUT_MOVE_SPEED

    if keys[pygame.K_UP]:
        runner.move_y = -GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_DOWN]:
        runner.move_y = GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_LEFT]:
        runner.move_x = -GameConfig.INPUT_MOVE_SPEED
    if keys[pygame.K_RIGHT]:
        runner.move_x = GameConfig.INPUT_MOVE_SPEED

def update_timer():
    global timer
    global runner_wins
    global draws
    global collection_count
    if(frame_count % SimConfig.FPS == 0 and frame_count != 0): # if 1 second has passed
        timer -= 1

    if(timer == 0): # if timer hits 0, reset
        # if collection_count >= GameConfig.MIN_RUNNER_WIN_COND:
        #     stats["runner_wins"] += 1
        # else:
        stats["draws"] += 1
        reset_game()

def reset_game():
    global timer
    global stats
    global collection_count
    global frame_count
    global generation_iteration

    random.seed(stats["generations"]*1000) # this ensures every sim gets an even playing field with its generation
    x_or_y = random.randint(0, 1)
    flip = random.randint(0, 1)
    chaser_start_x = 0
    chaser_start_y = 0

    if(x_or_y == 0): # x
        chaser_start_x = random.randint(int(GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH), int(SimConfig.SCREEN_WIDTH - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH))
        if flip == 0:
            chaser_start_y = GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT
        else:
            chaser_start_y = SimConfig.SCREEN_HEIGHT - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT
    else: # y
        chaser_start_y = random.randint(int(GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT), int(SimConfig.SCREEN_HEIGHT - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT))
        if flip == 0:
            chaser_start_x = GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH
        else:
            chaser_start_x = SimConfig.SCREEN_WIDTH - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH

    runner_start_x = SimConfig.SCREEN_WIDTH - chaser_start_x
    runner_start_y = SimConfig.SCREEN_HEIGHT - chaser_start_y

    chaser.x_pos = chaser_start_x
    chaser.y_pos = chaser_start_y
    runner.x_pos = runner_start_x
    runner.y_pos = runner_start_y
    timer = SimConfig.TIMER
    randomize_position(GameConfig.COLLECTIBLE_START_DIST_RUNNER)

    if SimConfig.ALLOW_RECORDINGS:
        check_recording(sim_done=True) # this tells it to save the recording if there is one

    stats["simulations"] += 1
    # print(f"Simulations: {stats['simulations']}", end="\r", flush=True)
    stats["total_collectibles"] += collection_count
    runner_fitness[generation_iteration] += NNConfig.RUNNER_FITNESS["collected"] * collection_count
    collection_count = 0
    frame_count = 0
    generation_iteration += 1
    if generation_iteration >= NNConfig.GENERERATION_SIZE: #
        reset_generation()


    with open(SimConfig.STATS_FILENAME, "w") as file:
        json.dump(stats, file, indent=4)


def detect_entity_collision():
    global collection_count
    global chaser_wins
    if(chaser.check_collision(runner)):
        stats["chaser_wins"] += 1
        chaser_fitness[generation_iteration] += NNConfig.CHASER_FITNESS["win"]
        # chaser_fitness[generation_iteration] += NNConfig.CHASER_FITNESS["extra_time"] * timer
        runner_fitness[generation_iteration] += NNConfig.RUNNER_FITNESS["loss"]
        reset_game()

    if(runner.check_collision(collectible)):
        # randomize_position(GameConfig.COLLECTIBLE_DIST_RUNNER)
        collection_count += 1
        stats["runner_wins"] += 1
        chaser_fitness[generation_iteration] += NNConfig.CHASER_FITNESS["loss"]
        reset_game()
        #print(collection_count)
    

def randomize_position(scale): # for collectible, find a position on the screen that isnt too close to the runner
    global runner_collectible_distance
    collectible.x_pos = random.randint(int(screen_width * 0.1), int(screen_width * 0.9))
    collectible.y_pos = random.randint(int(screen_height * 0.1), int(screen_height * 0.9))
    runner_collectible_distance = calc_shortest_dist(runner, collectible)["distance"]

    while runner_collectible_distance < screen_width*scale:
        collectible.x_pos = random.randint(int(screen_width * 0.1), int(screen_width * 0.9))
        collectible.y_pos = random.randint(int(screen_height * 0.1), int(screen_height * 0.9))
        runner_collectible_distance = calc_shortest_dist(runner, collectible)["distance"]

def display_timer():
    timer_text = font.render(f"Time: {timer}", True, SimConfig.TIMER_COLOR)
    screen.blit(timer_text, SimConfig.TIMER_LOC)

def wrap_delta(target, current, world_size): # helper function
    delta = target - current

    if delta > world_size / 2:
        delta -= world_size
    elif delta < -world_size / 2:
        delta += world_size

    return delta

def calc_shortest_dist(entity1, entity2):
    # line between entities
    line_x1 = entity1.x_pos + entity1.width/2
    line_y1 = entity1.y_pos + entity1.height/2

    line_x2 = entity2.x_pos + entity2.width/2
    line_y2 = entity2.y_pos + entity2.height/2

    dx = wrap_delta(line_x2, line_x1, screen_width)
    dy = wrap_delta(line_y2, line_y1, screen_height)

    distance = (dx**2 + dy**2)**0.5

    return { "line_x1": line_x1, "line_y1": line_y1, "line_x2": line_x2, "line_y2": line_y2, "dx": dx, "dy": dy, "distance": distance }

def calc_distances():
    global chaser_runner_dx
    global chaser_runner_dy
    global chaser_collectible_dx
    global chaser_collectible_dy
    global runner_collectible_dx
    global runner_collectible_dy

    global chaser_runner_distance
    global chaser_collectible_distance
    global runner_collectible_distance
    global past_chaser_runner_distance
    global past_chaser_collectible_distance
    global past_runner_collectible_distance

    past_chaser_runner_distance = chaser_runner_distance
    past_chaser_collectible_distance = chaser_collectible_distance
    past_runner_collectible_distance = runner_collectible_distance

    entity_dist_results = calc_shortest_dist(chaser, runner)
    chaser_dist_results = calc_shortest_dist(chaser, collectible)
    runner_dist_results = calc_shortest_dist(runner, collectible)

    chaser_runner_dx = entity_dist_results["dx"]
    chaser_runner_dy = entity_dist_results["dy"]
    chaser_runner_distance = entity_dist_results["distance"]

    chaser_collectible_dx = chaser_dist_results["dx"]
    chaser_collectible_dy = chaser_dist_results["dy"]
    chaser_collectible_distance = chaser_dist_results["distance"]

    runner_collectible_dx = runner_dist_results["dx"]
    runner_collectible_dy = runner_dist_results["dy"]
    runner_collectible_distance = runner_dist_results["distance"]

    #print(chaser_runner_dx, chaser_runner_dy, chaser_collectible_dx, chaser_collectible_dy, runner_collectible_dx, runner_collectible_dy)
    if frame_count == 0:
        past_chaser_runner_distance = chaser_runner_distance
        past_chaser_collectible_distance = chaser_collectible_distance
        past_runner_collectible_distance = runner_collectible_distance

def check_recording(sim_done = False):
    global recording_frame_data
    is_recording = False
    current_generation = stats["generations"] + 1

    if generation_iteration == 0:
        if current_generation == 1:
            is_recording = True
        elif current_generation % 5 == 0 and current_generation <= 50:
            is_recording = True
        elif current_generation % 25 == 0 and current_generation <= 500:
            is_recording = True
        elif current_generation % 250 == 0 and current_generation <= 2000:
            is_recording = True
        elif current_generation % 1000 == 0 and current_generation <= 10000:
            is_recording = True
        elif current_generation % 5000 == 0:
            is_recording = True

    if is_recording and not sim_done:
        recording_frame_data.append([
            frame_count + 1,
            chaser.x_pos,
            chaser.y_pos,
            runner.x_pos,
            runner.y_pos,
            collectible.x_pos,
            collectible.y_pos
        ])
    if is_recording and sim_done:
        filename = f"recordings/generation_{current_generation}.csv"

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "frame",
                "chaser_x",
                "chaser_y",
                "runner_x",
                "runner_y",
                "collectible_x",
                "collectible_y"
            ])

            writer.writerows(recording_frame_data)
        recording_frame_data = []

def update_replay():
    global timer
    if(frame_count < len(replay_frame_data) and not (replay_frame_data.loc[replay_frame_data["frame"] == frame_count + 1]).empty):
        chaser.x_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "chaser_x"].iloc[0]
        chaser.y_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "chaser_y"].iloc[0]

        runner.x_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "runner_x"].iloc[0]
        runner.y_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "runner_y"].iloc[0]

        collectible.x_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "collectible_x"].iloc[0]
        collectible.y_pos = replay_frame_data.loc[replay_frame_data["frame"] == frame_count+1, "collectible_y"].iloc[0]

        if(frame_count % SimConfig.FPS == 0 and frame_count != 0): # if 1 second has passed
            timer -= 1
        
def populate_generation(nn_template, survivors):
    generation = []
    for i in range(NNConfig.GENERERATION_SIZE):
        if i < NNConfig.SURVIVORS_SIZE: # save top survivors
            generation.append(copy.deepcopy(survivors[i]))
        elif i < NNConfig.SURVIVORS_SIZE * 2: # heavily mutate next few
            survivor_index = i % NNConfig.SURVIVORS_SIZE

            mutated_nn = copy.deepcopy(survivors[survivor_index])
            mutated_nn = nn_template.heavily_mutate(mutated_nn)

            generation.append(mutated_nn)
        else: # regular mutation for the rest
            survivor_index = i % NNConfig.SURVIVORS_SIZE

            mutated_nn = copy.deepcopy(survivors[survivor_index])
            mutated_nn = nn_template.mutate(mutated_nn)

            generation.append(mutated_nn)

    return generation

def draw_lines(entity1, entity2, color):
    distances = calc_shortest_dist(entity1, entity2)

    pygame.draw.line(screen, color, (distances["line_x1"], distances["line_y1"]), (distances["line_x1"] + distances["dx"], distances["line_y1"] + distances["dy"]), GameConfig.LINE_WIDTH)
    pygame.draw.line(screen, color, (distances["line_x2"], distances["line_y2"]), (distances["line_x2"] - distances["dx"], distances["line_y2"] - distances["dy"]), GameConfig.LINE_WIDTH)

def gather_input_nodes():
    # normalize values before output
    time = frame_count / (SimConfig.FPS * SimConfig.TIMER)

    c_r_dx = chaser_runner_dx / SimConfig.SCREEN_WIDTH
    c_r_dx_sin = math.sin(2 * math.pi * c_r_dx)
    c_r_dx_cos = math.cos(2 * math.pi * c_r_dx)

    c_r_dy = chaser_runner_dy / SimConfig.SCREEN_HEIGHT
    c_r_dy_sin = math.sin(2 * math.pi * c_r_dy)
    c_r_dy_cos = math.cos(2 * math.pi * c_r_dy)

    c_r_dist = chaser_runner_distance / screen_diagonal

    c_c_dx = chaser_collectible_dx / SimConfig.SCREEN_WIDTH
    c_c_dx_sin = math.sin(2 * math.pi * c_c_dx)
    c_c_dx_cos = math.cos(2 * math.pi * c_c_dx)

    c_c_dy = chaser_collectible_dy / SimConfig.SCREEN_HEIGHT
    c_c_dy_sin = math.sin(2 * math.pi * c_c_dy)
    c_c_dy_cos = math.cos(2 * math.pi * c_c_dy)

    c_c_dist = chaser_collectible_distance / screen_diagonal

    r_c_dx = runner_collectible_dx / SimConfig.SCREEN_WIDTH
    r_c_dx_sin = math.sin(2 * math.pi * r_c_dx)
    r_c_dx_cos = math.cos(2 * math.pi * r_c_dx)

    r_c_dy = runner_collectible_dy / SimConfig.SCREEN_HEIGHT
    r_c_dy_sin = math.sin(2 * math.pi * r_c_dy)
    r_c_dy_cos = math.cos(2 * math.pi * r_c_dy)

    r_c_dist = runner_collectible_distance / screen_diagonal

    c_m_x = chaser.move_x / GameConfig.CHASER_MAX_SPEED
    c_m_y = chaser.move_y / GameConfig.CHASER_MAX_SPEED

    r_m_x = runner.move_x / GameConfig.RUNNER_MAX_SPEED
    r_m_y = runner.move_y / GameConfig.RUNNER_MAX_SPEED

    return [ time, c_r_dx_sin, c_r_dx_cos, c_r_dy_sin, c_r_dy_cos, c_r_dist, c_c_dx_sin, c_c_dx_cos, c_c_dy_sin, c_c_dy_cos, c_c_dist, r_c_dx_sin, r_c_dx_cos, r_c_dy_sin, r_c_dy_cos, r_c_dist, c_m_x, c_m_y, r_m_x, r_m_y ]

def reset_generation():
    global chaser_fitness
    global runner_fitness
    global chaser_survivors
    global runner_survivors
    global chaser_generation
    global runner_generation

    global generation_iteration

    top_chasers = []
    top_chasers_score = []
    for i in range(NNConfig.GENERERATION_SIZE):
        if i < NNConfig.SURVIVORS_SIZE:
            top_chasers.append(chaser_generation[i])
            top_chasers_score.append(chaser_fitness[i])
        else:
            for j in range(NNConfig.SURVIVORS_SIZE):
                if chaser_fitness[i] > top_chasers_score[j]:
                    if j == 0:
                        for k in range(NNConfig.SURVIVORS_SIZE - 1, j, -1):
                            top_chasers[k] = top_chasers[k - 1]
                            top_chasers_score[k] = top_chasers_score[k - 1]

                        top_chasers[j] = chaser_generation[i]
                        top_chasers_score[j] = chaser_fitness[i]

                        with open(NNConfig.CHASER_NN_FILENAME, "w") as file:
                            json.dump(top_chasers[0], file, indent=4)

                        if chaser_fitness[i] > stats["chaser_best_score"]:
                            stats["chaser_best_score"] = top_chasers_score[0]
                            
                        break
                    else:
                        for k in range(NNConfig.SURVIVORS_SIZE - 1, j, -1):
                            top_chasers[k] = top_chasers[k - 1]
                            top_chasers_score[k] = top_chasers_score[k - 1]

                        top_chasers[j] = chaser_generation[i]
                        top_chasers_score[j] = chaser_fitness[i]
                        break

    chaser_survivors = top_chasers
    stats["generations"] += 1
    random.seed(stats["generations"]*1000)



    top_runners = []
    top_runners_score = []
    for i in range(NNConfig.GENERERATION_SIZE):
        if i < NNConfig.SURVIVORS_SIZE:
            top_runners.append(runner_generation[i])
            top_runners_score.append(runner_fitness[i])
        else:
            for j in range(NNConfig.SURVIVORS_SIZE):
                if runner_fitness[i] > top_runners_score[j]:
                    if j == 0:
                        for k in range(NNConfig.SURVIVORS_SIZE - 1, j, -1):
                            top_runners[k] = top_runners[k - 1]
                            top_runners_score[k] = top_runners_score[k - 1]

                        top_runners[j] = runner_generation[i]
                        top_runners_score[j] = runner_fitness[i]

                        with open(NNConfig.RUNNER_NN_FILENAME, "w") as file:
                            json.dump(top_runners[0], file, indent=4)

                        if runner_fitness[i] > stats["runner_best_score"]:
                            stats["runner_best_score"] = top_runners_score[0]

                        break
                    else:
                        for k in range(NNConfig.SURVIVORS_SIZE - 1, j, -1):
                            top_runners[k] = top_runners[k - 1]
                            top_runners_score[k] = top_runners_score[k - 1]

                        top_runners[j] = runner_generation[i]
                        top_runners_score[j] = runner_fitness[i]
                        break

    runner_survivors = top_runners

    chaser_generation = populate_generation(chaser_nn, chaser_survivors)
    runner_generation = populate_generation(runner_nn, runner_survivors)

    chaser_fitness = [0] * NNConfig.GENERERATION_SIZE
    runner_fitness = [0] * NNConfig.GENERERATION_SIZE
    generation_iteration = 0


def calculate_fitness():
    global chaser_fitness
    global runner_fitness

    chaser_fitness[generation_iteration] += NNConfig.CHASER_FITNESS["stalling"]
    runner_fitness[generation_iteration] += NNConfig.RUNNER_FITNESS["stalling"]

    chaser_runner_delta = past_chaser_runner_distance - chaser_runner_distance
    chaser_collectible_delta = past_chaser_collectible_distance - chaser_collectible_distance
    runner_collectible_delta = past_runner_collectible_distance - runner_collectible_distance

    chaser_fitness[generation_iteration] += chaser_runner_delta * NNConfig.CHASER_FITNESS["runner_dist_sec"]
    runner_fitness[generation_iteration] += -chaser_runner_delta * NNConfig.RUNNER_FITNESS["chaser_dist_sec"]

    runner_fitness[generation_iteration] += runner_collectible_delta * NNConfig.RUNNER_FITNESS["collectible_dist_sec"]

    # if runner_collectible_distance <= screen_diagonal * NNConfig.RUNNER_MIN_DIST_COLLECTIBLE:
    # temporarily got rid of this
    # only reward chaser for being close to the collectible if the runner is also close to it
    chaser_fitness[generation_iteration] += chaser_collectible_delta * NNConfig.CHASER_FITNESS["collectible_dist_sec"]

def calculate_movement():
    # apply friction from last frame
    chaser.move_x *= GameConfig.ACCELERATION_DAMPENING
    chaser.move_y *= GameConfig.ACCELERATION_DAMPENING
    runner.move_x *= GameConfig.ACCELERATION_DAMPENING
    runner.move_y *= GameConfig.ACCELERATION_DAMPENING

    input_nodes = gather_input_nodes()

    chaser_decision = chaser_nn.make_decision(chaser_generation[generation_iteration], input_nodes)
    dx = NNConfig.CHASER_OUTPUTS[chaser_decision]["move_x"]
    dy = NNConfig.CHASER_OUTPUTS[chaser_decision]["move_y"]

    chaser.move_x += dx
    chaser.move_y += dy

    speed = math.sqrt(chaser.move_x**2 + chaser.move_y**2)
    if speed > GameConfig.CHASER_MAX_SPEED:
        scale = GameConfig.CHASER_MAX_SPEED / speed
        chaser.move_x *= scale
        chaser.move_y *= scale

    runner_decision = runner_nn.make_decision(runner_generation[generation_iteration], input_nodes)
    dx = NNConfig.RUNNER_OUTPUTS[runner_decision]["move_x"]
    dy = NNConfig.RUNNER_OUTPUTS[runner_decision]["move_y"]

    runner.move_x += dx
    runner.move_y += dy

    speed = math.sqrt(runner.move_x**2 + runner.move_y**2)
    if speed > GameConfig.RUNNER_MAX_SPEED:
        scale = GameConfig.RUNNER_MAX_SPEED / speed
        runner.move_x *= scale
        runner.move_y *= scale

    chaser.move()
    runner.move()

def update():
    if SimConfig.REPLAY_MODE and not SimConfig.SIM_MODE:
        update_replay()
    else:
        update_timer()

        calc_distances()

        calculate_fitness()

        if SimConfig.ALLOW_RECORDINGS:
            check_recording()

        if SimConfig.INPUTS_ALLOWED and not SimConfig.SIM_MODE:
            check_inputs()

        calculate_movement()
        
        detect_entity_collision()


def draw():
    screen.fill(GameConfig.BACKGROUND_COLOR) 

    collectible.draw(screen)
    runner.draw(screen)
    chaser.draw(screen)

    draw_lines(chaser, runner, GameConfig.ENTITYxENTITY_LINE_COLOR)
    draw_lines(chaser, collectible, GameConfig.CHASERxCOLLECTIBLE_LINE_COLOR)
    draw_lines(runner, collectible, GameConfig.RUNNERxCOLLECTIBLE_LINE_COLOR)

    display_timer()

# ================================================== END FUNCTIONS ===================================================

screen_width = SimConfig.SCREEN_WIDTH
screen_height = SimConfig.SCREEN_HEIGHT
screen_diagonal = (SimConfig.SCREEN_WIDTH**2 + SimConfig.SCREEN_HEIGHT**2)**0.5

frame_count = 0

# read all time stats from json
default_stats = {
    "simulations": 0,
    "generations": 0,
    "chaser_wins": 0,
    "runner_wins": 0,
    "draws": 0,
    "total_collectibles": 0,
    "chaser_best_score": 0,
    "runner_best_score": 0
}

if os.path.exists(SimConfig.STATS_FILENAME):
    try:
        with open(SimConfig.STATS_FILENAME, "r") as file:
            stats = json.load(file)
    except (json.JSONDecodeError, IOError):
        stats = default_stats.copy()
else:
    stats = default_stats.copy()

# ensure file exists going forward
with open(SimConfig.STATS_FILENAME, "w") as file:
    json.dump(stats, file, indent=4)

# gathers information about every frame during recorded sims
recording_frame_data = []

# extracts recording information from csv for replay
replay_frame_data = []
if SimConfig.REPLAY_MODE and not SimConfig.SIM_MODE:
    replay_number = input("\nWelcome to Replay Mode! Please enter the relative path of the replay you want to view. \nReplay path: ")
    try:
        replay_frame_data = pd.read_csv(f"{replay_number}")
    except:
        sys.exit("Session Aborted. A replay with that path does not exist.")

timer = SimConfig.TIMER

chaser_runner_dx = 0
chaser_runner_dy = 0

chaser_collectible_dx = 0
chaser_collectible_dy = 0

runner_collectible_dx = 0
runner_collectible_dy = 0

chaser_runner_distance = 0
chaser_collectible_distance = 0
runner_collectible_distance = 0

past_chaser_runner_distance = 0
past_chaser_collectible_distance = 0
past_runner_collectible_distance = 0

chaser_nn = NeuralNetwork(NNConfig.CHASER_NN_FILENAME, NNConfig.DEF_STRUCTURE)
runner_nn = NeuralNetwork(NNConfig.RUNNER_NN_FILENAME, NNConfig.DEF_STRUCTURE)

with open(NNConfig.CHASER_NN_FILENAME, "r") as file:
    chaser_alpha = json.load(file)
chaser_survivors = [chaser_alpha] * NNConfig.SURVIVORS_SIZE
chaser_generation = populate_generation(chaser_nn, chaser_survivors)
chaser_fitness = [0] * NNConfig.GENERERATION_SIZE

with open(NNConfig.RUNNER_NN_FILENAME, "r") as file:
    runner_alpha = json.load(file)
runner_survivors = [runner_alpha] * NNConfig.SURVIVORS_SIZE
runner_generation = populate_generation(runner_nn, runner_survivors)
runner_fitness = [0] * NNConfig.GENERERATION_SIZE

random.seed(stats["generations"]*1000) # this ensures every sim gets an even playing field with its generation
x_or_y = random.randint(0, 1)
flip = random.randint(0, 1)
chaser_start_x = 0
chaser_start_y = 0

if(x_or_y == 0): # x
    chaser_start_x = random.randint(int(GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH), int(SimConfig.SCREEN_WIDTH - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH))
    if flip == 0:
        chaser_start_y = GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT
    else:
        chaser_start_y = SimConfig.SCREEN_HEIGHT - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT
else: # y
    chaser_start_y = random.randint(int(GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT), int(SimConfig.SCREEN_HEIGHT - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_HEIGHT))
    if flip == 0:
        chaser_start_x = GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH
    else:
        chaser_start_x = SimConfig.SCREEN_WIDTH - GameConfig.START_POS_TRIMMING*SimConfig.SCREEN_WIDTH

runner_start_x = SimConfig.SCREEN_WIDTH - chaser_start_x
runner_start_y = SimConfig.SCREEN_HEIGHT - chaser_start_y

chaser = Entity(chaser_start_x, chaser_start_y, GameConfig.CHASER_COLOR, chaser_nn)
runner = Entity(runner_start_x, runner_start_y, GameConfig.RUNNER_COLOR, runner_nn)
collectible = Entity(0, 0, GameConfig.COLLECTIBLE_COLOR, width=25, height=25)
randomize_position(GameConfig.COLLECTIBLE_START_DIST_RUNNER)
collection_count = 0


pygame.init()

if(SimConfig.SIM_MODE):
    clock = None
    pygame.display.set_caption("Simulating - " + SimConfig.WINDOW_NAME)
    print("\nSimulating... Press ESC to stop. DO NOT USE CRTL+C")
else:
    pygame.display.set_caption(SimConfig.WINDOW_NAME)
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, SimConfig.TIMER_SIZE)

screen = pygame.display.set_mode((screen_width, screen_height))
running = True
generation_iteration = 0

# =============================================== GAME LOOP ==============================================

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            if SimConfig.SIM_MODE:
                print("\nShutdown simulation")
        if event.type == pygame.KEYDOWN and SimConfig.SIM_MODE:
            if event.key == pygame.K_ESCAPE:
                running = False
                print("\nShutdown simulation")

    update()

    if not SimConfig.SIM_MODE:
        draw()

        pygame.display.flip()
        clock.tick(SimConfig.FPS)

    frame_count += 1
    # print(f"Frame Count: {frame_count}", end="\r", flush=True)

pygame.quit()
# put save functions here if needed