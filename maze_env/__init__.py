from gym.envs.registration import register

register(
    id='Maze-v0',
    entry_point='maze_env.Maze:MazeEnv'
)
