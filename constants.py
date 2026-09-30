# A centralized location for all core variables to potentially change later

class EntityConfig:
    DEF_WIDTH = 50
    DEF_HEIGHT = 50
    COLLISION_TRIMMING = 3
    WALL_TRIMMING = 25


class NNConfig:
    DEF_INPUT_NODES = 20
    DEF_OUTPUT_NODES = 9
    DEF_STRUCTURE = [16, 16]
    SEED_WEIGHT_MAX = 0.5

    CHASER_NN_FILENAME = "chaser_nn.json"
    RUNNER_NN_FILENAME = "runner_nn.json"

    GENERERATION_SIZE = 50
    SURVIVORS_SIZE = 5 # 1 will always be the best alltime survivor

    WEIGHT_MUTATION_CHANCE = 10 # 1/10 per weight
    BIAS_MUTATION_CHANCE = 20 # 1/20 per bias
    WEIGHT_MUTATION_STRENGTH = 0.2
    BIAS_MUTATION_STRENGTH = 0.1
    HEAVY_MUTATION_CONSISTENCY = 0.5 # times 2 amount of weights/biases mutated
    HEAVY_MUTATION_STRENGTH = 2 # times 2 strength

    CHASER_FITNESS = {"win": 1000, "loss": -1000, "runner_dist_sec": 0.3, "collectible_dist_sec": 0.3, "extra_time": 40, "warp": -3, "stalling": -0.1}
    CHASER_MIN_DIST_RUNNER = 0.25 # 1/4 of the screen
    CHASER_MIN_DIST_COLLECTIBLE = 0.25 # 1/4 of the screen
    CHASER_OUTPUTS = [{"move_x": 0, "move_y": 0}, {"move_x": 1, "move_y": 0}, {"move_x": 1, "move_y": 1}, {"move_x": 0, "move_y": 1}, {"move_x": -1, "move_y": 1}, {"move_x": -1, "move_y": 0}, {"move_x": -1, "move_y": -1}, {"move_x": 0, "move_y": -1}, {"move_x": 1, "move_y": -1}]

    RUNNER_FITNESS = {"collected": 1000, "chaser_dist_sec": 0.3, "collectible_dist_sec": 0.3, "loss": -1000, "warp": -3, "stalling": -0.1}
    RUNNER_MIN_DIST_CHASER = 0.5 # 1/2 of the screen
    RUNNER_MIN_DIST_COLLECTIBLE = 0.1 # 1/10 of the screen
    RUNNER_OUTPUTS = [{"move_x": 0, "move_y": 0}, {"move_x": 1, "move_y": 0}, {"move_x": 1, "move_y": 1}, {"move_x": 0, "move_y": 1}, {"move_x": -1, "move_y": 1}, {"move_x": -1, "move_y": 0}, {"move_x": -1, "move_y": -1}, {"move_x": 1, "move_y": -1}, {"move_x": 1, "move_y": -1}]


class SimConfig:
    SIM_MODE = False # designed to run as fast as possible. No drawing, no sleep()
    REPLAY_MODE = False # used to replay past simulations. Recordings are in the recordings folder
    ALLOW_RECORDINGS = True # will record some simulations for the sake of replaying NN progress
    INPUTS_ALLOWED = False # also turns off learning when True. When a user is playing, they just vs the current best ai model
    FPS = 60
    TIMER = 30 # seconds for each sim
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 500
    WINDOW_NAME = "AI Chase"
    TIMER_SIZE = 48 # font size
    TIMER_COLOR = (255, 255, 255)
    TIMER_LOC = (SCREEN_WIDTH - 150, 10)

    STATS_FILENAME = "stats.json"


class GameConfig:
    INPUT_MOVE_SPEED = 5

    BACKGROUND_COLOR = (140, 200, 230)

    LINE_WIDTH = 2
    ENTITYxENTITY_LINE_COLOR = (230, 230, 0)
    CHASERxCOLLECTIBLE_LINE_COLOR = (150, 100, 0)
    RUNNERxCOLLECTIBLE_LINE_COLOR = (0, 100, 150)

    START_POS_TRIMMING = 0.2

    ACCELERATION_DAMPENING = 0.9 # slows movement if not continuously supported
    
    CHASER_COLOR = (255, 0, 0)
    CHASER_MAX_SPEED = 4

    RUNNER_COLOR = (50, 0, 200)
    RUNNER_MAX_SPEED = 4

    MIN_RUNNER_WIN_COND = 1 # the minimum collectibles a runner needs for it to be considered a win
    
    COLLECTIBLE_COLOR = (0, 255, 0)

    # numbers to put into a randomize function. decides how far to place the collectible from the runner
    COLLECTIBLE_START_DIST_RUNNER = 0.5
    COLLECTIBLE_DIST_RUNNER = 0.25