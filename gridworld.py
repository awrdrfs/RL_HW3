import numpy as np
import random

class GridWorld:
    def __init__(self, mode='static', size=4):
        self.size = size
        self.mode = mode
        self.reset()

    def reset(self):
        # 0: empty, 1: player, 2: goal, 3: pit, 4: wall
        self.grid = np.zeros((self.size, self.size))
        
        if self.mode == 'static':
            self.goal = (0, 0)
            self.pit = (0, 1)
            self.wall = (1, 1)
            self.player = (0, 3)
        elif self.mode == 'player':
            self.goal = (0, 0)
            self.pit = (0, 1)
            self.wall = (1, 1)
            self.player = self._get_random_pos(exclude=[self.goal, self.pit, self.wall])
        elif self.mode == 'random':
            positions = random.sample([(r, c) for r in range(self.size) for c in range(self.size)], 4)
            self.goal = positions[0]
            self.pit = positions[1]
            self.wall = positions[2]
            self.player = positions[3]
            
        self._update_grid()
        return self.get_state()

    def _get_random_pos(self, exclude=[]):
        while True:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos not in exclude:
                return pos

    def _update_grid(self):
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.goal] = 2
        self.grid[self.pit] = 3
        self.grid[self.wall] = 4
        self.grid[self.player] = 1

    def get_state(self):
        # 這裡返回 flatten 的 grid 作為狀態，或者也可以返回座標
        # 為了 DQN，我們可以使用 one-hot 或者簡單的數值
        # 這裡先使用簡單的數值表示並 flatten
        return self.grid.flatten()

    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        r, c = self.player
        if action == 0: # up
            new_pos = (max(0, r-1), c)
        elif action == 1: # down
            new_pos = (min(self.size-1, r+1), c)
        elif action == 2: # left
            new_pos = (r, max(0, c-1))
        elif action == 3: # right
            new_pos = (r, min(self.size-1, c+1))
        else:
            raise ValueError("Invalid action")

        # Check if wall
        if new_pos == self.wall:
            new_pos = (r, c) # Stay put
            
        self.player = new_pos
        self._update_grid()
        
        # Reward and Done
        reward = -0.01 # step penalty
        done = False
        
        if self.player == self.goal:
            reward = 1.0
            done = True
        elif self.player == self.pit:
            reward = -1.0
            done = True
            
        return self.get_state(), reward, done

    def render(self):
        char_map = {0: '.', 1: 'P', 2: 'G', 3: 'X', 4: 'W'}
        for r in range(self.size):
            line = ""
            for c in range(self.size):
                line += char_map[self.grid[r, c]] + " "
            print(line)
        print()
