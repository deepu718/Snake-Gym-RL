# SnakeWorld Example Design

# 🎯 Skill: should be able to go to the food without colliding with the body and walls

# 👀 Information: agent position body possition food position in the environment

# 🎮 Actions: Move up, down, left, or right without colliding with itself

# 🏆 Success: getting a high score with eating food

# ⏰ End: when full  environment is filled by snake


from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import namedtuple, deque

class Actions(Enum):
    right = 0
    straight = 1
    left = 2

class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, size=5, obs_type="vector"):
        self.size = size
        self.window_size = 512
        self.window = None
        self.clock = None
        self.render_mode = render_mode
        self.obs_type = obs_type

        self._head_location = None
        self._food_location = None
        self._snake_body = []

        
        if self.obs_type == "grid":
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(self.size, self.size, 1), 
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            )

        # RELATIVE ACTION SPACE
        self.action_space = spaces.Discrete(3)

        self.direction = Directions.RIGHT
        self._dir_to_vec = {
            Directions.UP:    np.array([0, -1]),
            Directions.RIGHT: np.array([1,  0]),
            Directions.DOWN:  np.array([0,  1]),
            Directions.LEFT:  np.array([-1, 0]),
        }

    def _update_direction(self, action):
        if action == Actions.straight.value:
            return self.direction
        elif action == Actions.right.value:
            return Directions((self.direction.value + 1) % 4)
        elif action == Actions.left.value:
            return Directions((self.direction.value - 1) % 4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._head_location = np.array([self.size//2,self.size//2])
        self._snake_body = [tuple(self._head_location), (self._head_location[0]-1,self._head_location[1]), (self._head_location[0]-2,self._head_location[1])]

        self._food_location = self._head_location
        while tuple(self._food_location) in self._snake_body:
            self._food_location =  self.np_random.integers(0, self.size, size=2, dtype=int)
        self.direction = Directions.RIGHT

        self.score = 0
        

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def _get_obs(self):
        if self.obs_type == "grid":
            return self._get_grid_obs()
        else:
            return self._get_vector_obs()

    
    def _get_grid_obs(self):
        grid = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        
        
        for segment in self._snake_body:
            x, y = segment
            if 0 <= x < self.size and 0 <= y < self.size:
                grid[x, y, 0] = 100

        
        hx, hy = self._head_location
        if 0 <= hx < self.size and 0 <= hy < self.size:
            grid[hx, hy, 0] = 180

        
        fx, fy = self._food_location
        if 0 <= fx < self.size and 0 <= fy < self.size:
            grid[fx, fy, 0] = 255
            
        return grid

    def _get_vector_obs(self):
        head = self._head_location
        
        vec_straight = self._dir_to_vec[self.direction]
        dir_right_enum = Directions((self.direction.value + 1) % 4)
        vec_right = self._dir_to_vec[dir_right_enum]
        dir_left_enum = Directions((self.direction.value - 1) % 4)
        vec_left = self._dir_to_vec[dir_left_enum]

        dist_s = self.get_distance_to_collision(head, vec_straight)
        dist_r = self.get_distance_to_collision(head, vec_right)
        dist_l = self.get_distance_to_collision(head, vec_left)

        point_straight = head + vec_straight
        point_right    = head + vec_right
        point_left     = head + vec_left
        
        freedom_s = self.get_accessible_area_fraction(point_straight)
        freedom_r = self.get_accessible_area_fraction(point_right)
        freedom_l = self.get_accessible_area_fraction(point_left)

        dx = self._food_location[0] - head[0]
        dy = self._food_location[1] - head[1]

        food_front, food_right, food_left, food_behind = False, False, False, False
        if self.direction == Directions.RIGHT:
            food_front, food_behind = dx > 0, dx < 0
            food_right, food_left   = dy > 0, dy < 0
        elif self.direction == Directions.LEFT:
            food_front, food_behind = dx < 0, dx > 0
            food_right, food_left   = dy < 0, dy > 0
        elif self.direction == Directions.UP:
            food_front, food_behind = dy < 0, dy > 0
            food_right, food_left   = dx > 0, dx < 0
        elif self.direction == Directions.DOWN:
            food_front, food_behind = dy > 0, dy < 0
            food_right, food_left   = dx < 0, dx > 0

        state = [
            1.0 / dist_s, 1.0 / dist_r, 1.0 / dist_l,
            freedom_s, freedom_r, freedom_l,
            int(food_front), int(food_right), int(food_left), int(food_behind)
        ]
        return np.array(state, dtype=np.float32)

    def get_distance_to_collision(self, start_pos, direction_vec):
        distance = 0
        current_pos = start_pos.copy()
        while True:
            current_pos = current_pos + direction_vec
            distance += 1
            if (current_pos[0] < 0 or current_pos[0] >= self.size or 
                current_pos[1] < 0 or current_pos[1] >= self.size):
                return distance
            if tuple(current_pos) in self._snake_body:
                return distance
            if distance > self.size * 2: return distance

    def get_accessible_area_fraction(self, start_pos):
        if (start_pos[0] < 0 or start_pos[0] >= self.size or 
            start_pos[1] < 0 or start_pos[1] >= self.size or 
            tuple(start_pos) in self._snake_body[:-1]):
            return 0.0
        queue = deque([tuple(start_pos)])
        visited = set([tuple(start_pos)])
        count = 0
        body_set = set(self._snake_body[:-1])
        while queue:
            cx, cy = queue.popleft()
            count += 1
            total_empty_squares = (self.size * self.size) - len(self._snake_body)
            if count >= total_empty_squares * 0.8: return 1.0
            for move in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + move[0], cy + move[1]
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    (nx, ny) not in visited and (nx, ny) not in body_set):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        total_empty_squares = max(1, (self.size * self.size) - len(self._snake_body))
        return count / total_empty_squares
    
    def _get_info(self):
        return {
            "score": self.score,
            "snake_length": len(self._snake_body),
            "food_dist": np.linalg.norm(self._head_location - self._food_location, ord=1),
            "head_loc": tuple(self._head_location),
            "food_loc": tuple(self._food_location)
        }

    def step(self, action):
        reward = -0.01
        terminated = False
        self.direction = self._update_direction(action)
        mov_vector = self._dir_to_vec[self.direction] 
        self._new_head_location = self._head_location + mov_vector

        # Wall Collision
        if np.any(self._new_head_location >= self.size) or np.any(self._new_head_location <= -1):
            reward = -10
            terminated = True
            observation = self._get_obs() # Calls dispatcher
            return observation, reward, terminated, False, self._get_info()

        self._snake_body.insert(0, tuple(self._new_head_location))
        self._head_location = self._new_head_location

        # Eat Food
        if not (np.array_equal(self._new_head_location, self._food_location)):
            self._snake_body.pop()
        else: 
            reward = 5
            self.score += 1
            if len(self._snake_body) == self.size * self.size:
                reward = 100  
                terminated = True
                observation = self._get_obs()
                return observation, reward, terminated, False, self._get_info()
            
            while tuple(self._food_location) in self._snake_body:
               self._food_location =  self.np_random.integers(0, self.size, size=2, dtype=int)

        # Self Collision
        if tuple(self._head_location) in self._snake_body[1:]: 
            reward = -10
            terminated = True
            observation = self._get_obs()
            return observation, reward, terminated, False, self._get_info()
        
        observation = self._get_obs()
        return observation, reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array": return self._render_frame()
        elif self.render_mode == "human": self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake DQN")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = self.window_size / self.size
        
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pix_square_size * self._food_location[0], pix_square_size * self._food_location[1], pix_square_size, pix_square_size))
        for i, segment in enumerate(self._snake_body):
            color = (0, 255, 0) if i > 0 else (0, 200, 0)
            pygame.draw.rect(canvas, color, pygame.Rect(pix_square_size * segment[0], pix_square_size * segment[1], pix_square_size, pix_square_size))
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (50, 50, 50), (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=1)
            pygame.draw.line(canvas, (50, 50, 50), (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=1)

        if self.render_mode == "human":
            pygame.display.set_caption(f"Snake DQN | Score: {self.score}")
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    # Set to None to test "Headless" mode
    env = SnakeWorldEnv(render_mode="human", size=10) 
    obs, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # This line will do nothing (won't crash, just skips)
        env.render() 
        
        # Print so you know it's actually running
        print(f"Step {i}: Reward {reward}, Terminated {terminated}")
        
        if terminated:
            env.reset()
            
    print("Done!")
    env.close()