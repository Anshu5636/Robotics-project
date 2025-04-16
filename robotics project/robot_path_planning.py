import numpy as np
import matplotlib.pyplot as plt
import pygame
import pygame.gfxdraw
import time
import sys
import random
from queue import PriorityQueue
import pandas as pd
import math

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
CYAN = (0, 255, 255)

# Constants
GRID_SIZE = 30  # Size of each cell in pixels
WINDOW_WIDTH = 900  # Window width in pixels
WINDOW_HEIGHT = 600  # Window height in pixels
ROWS = WINDOW_HEIGHT // GRID_SIZE
COLS = WINDOW_WIDTH // GRID_SIZE

class Robot:
    def __init__(self, x, y, radius=0.5, max_speed=1.0, max_accel=0.5):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.radius = radius
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.path_history = [self.position.copy()]

    def move(self, target_velocity, dt):
        # Simple kinematic model with acceleration limits
        desired_velocity_change = target_velocity - self.velocity
        actual_velocity_change = np.clip(
            desired_velocity_change,
            -self.max_accel * dt,
            self.max_accel * dt
        )

        self.velocity += actual_velocity_change

        # Apply speed limit
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed and speed > 0:
            self.velocity = self.velocity * self.max_speed / speed

        # Update position
        self.position += self.velocity * dt
        self.path_history.append(self.position.copy())

    def distance_to(self, point):
        return np.linalg.norm(self.position - point)

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, position):
        x, y = position
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x * GRID_SIZE, self.y * GRID_SIZE,
                                         self.width * GRID_SIZE, self.height * GRID_SIZE))

class CircleObstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def contains(self, position):
        x, y = position
        return np.sqrt((x - self.x)**2 + (y - self.y)**2) <= self.radius

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (int(self.x * GRID_SIZE), int(self.y * GRID_SIZE)),
                          int(self.radius * GRID_SIZE))

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self.start = None
        self.goal = None

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def set_start(self, x, y):
        self.start = np.array([x, y], dtype=float)

    def set_goal(self, x, y):
        self.goal = np.array([x, y], dtype=float)

    def is_valid_position(self, position):
        x, y = position
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        for obs in self.obstacles:
            if obs.contains(position):
                return False

        return True

    def random_valid_position(self):
        while True:
            pos = np.array([
                np.random.uniform(0, self.width),
                np.random.uniform(0, self.height)
            ])
            if self.is_valid_position(pos):
                return pos

    def draw(self, screen):
        # Draw background
        screen.fill(WHITE)

        # Draw grid lines
        for i in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GREY, (i, 0), (i, WINDOW_HEIGHT), 1)
        for i in range(0, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, GREY, (0, i), (WINDOW_WIDTH, i), 1)

        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        # Draw start and goal
        if self.start is not None:
            pygame.draw.circle(screen, GREEN,
                              (int(self.start[0] * GRID_SIZE), int(self.start[1] * GRID_SIZE)),
                              int(GRID_SIZE/2))

        if self.goal is not None:
            pygame.draw.circle(screen, RED,
                              (int(self.goal[0] * GRID_SIZE), int(self.goal[1] * GRID_SIZE)),
                              int(GRID_SIZE/2))

# Helper functions for algorithms
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(node, environment, step_size=0.5):
    """Get valid neighboring nodes in 8 directions"""
    x, y = node
    neighbors = []

    # 8 directions
    directions = [
        (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
        (step_size, step_size), (-step_size, step_size),
        (step_size, -step_size), (-step_size, -step_size)
    ]

    for dx, dy in directions:
        new_pos = np.array([x + dx, y + dy])
        if environment.is_valid_position(new_pos):
            neighbors.append(tuple(new_pos))

    return neighbors

def is_path_valid(environment, start, end):
    """Check if a straight path between start and end is collision-free"""
    direction = end - start
    distance = np.linalg.norm(direction)

    if distance == 0:
        return True

    direction = direction / distance  # Normalize

    # Check multiple points along the path
    num_checks = int(distance * 10)  # Adjust density of checks as needed
    for i in range(1, num_checks + 1):
        point = start + direction * (i * distance / num_checks)
        if not environment.is_valid_position(point):
            return False

    return True

def reconstruct_path(came_from, current):
    """Reconstruct path from came_from dictionary"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # Reverse to get start-to-goal order

def extract_path(tree, start, goal):
    """Extract path from RRT tree"""
    path = [goal]
    current = tuple(goal)

    while current != tuple(start):
        current = tree[current]
        path.append(np.array(current))

    return path[::-1]  # Reverse to get start-to-goal order

# Pathfinding Algorithms
def astar(environment, start, goal, heuristic_func=euclidean_distance):
    """A* Pathfinding Algorithm"""
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)

    open_set = PriorityQueue()
    open_set.put((0, start_tuple))
    came_from = {}
    g_score = {start_tuple: 0}
    f_score = {start_tuple: heuristic_func(start, goal)}

    open_set_hash = {start_tuple}
    explored_nodes = []

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)
        explored_nodes.append(current)

        if current == goal_tuple:
            path = reconstruct_path(came_from, current)
            return path, explored_nodes

        for neighbor in get_neighbors(current, environment):
            tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)

                if neighbor not in open_set_hash:
                    open_set.put((f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return None, explored_nodes  # No path found

def dijkstra(environment, start, goal):
    """Dijkstra's Algorithm"""
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)

    open_set = PriorityQueue()
    open_set.put((0, start_tuple))
    came_from = {}
    cost_so_far = {start_tuple: 0}

    open_set_hash = {start_tuple}
    explored_nodes = []

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)
        explored_nodes.append(current)

        if current == goal_tuple:
            path = reconstruct_path(came_from, current)
            return path, explored_nodes

        for neighbor in get_neighbors(current, environment):
            new_cost = cost_so_far[current] + euclidean_distance(current, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                if neighbor not in open_set_hash:
                    open_set.put((new_cost, neighbor))
                    open_set_hash.add(neighbor)

    return None, explored_nodes  # No path found

def rrt(environment, start, goal, max_iterations=1000, step_size=0.5, goal_sample_rate=0.05):
    """RRT (Rapidly-exploring Random Tree) Algorithm"""
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)

    tree = {start_tuple: None}  # node -> parent
    nodes = [start]
    explored_nodes = []

    for i in range(max_iterations):
        # Random sampling with goal bias
        if np.random.random() < goal_sample_rate:  # chance to sample the goal
            random_point = goal
        else:
            random_point = environment.random_valid_position()

        explored_nodes.append(tuple(random_point))

        # Find nearest node
        nearest_node = min(nodes, key=lambda n: euclidean_distance(n, random_point))
        nearest_tuple = tuple(nearest_node)

        # Steer towards random point
        direction = random_point - nearest_node
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance  # normalize

        new_node = nearest_node + direction * min(step_size, distance)
        new_tuple = tuple(new_node)

        # Check if valid
        if environment.is_valid_position(new_node):
            # Check if path is collision-free
            if is_path_valid(environment, nearest_node, new_node):
                nodes.append(new_node)
                tree[new_tuple] = nearest_tuple

                # Check if we can connect to goal
                if euclidean_distance(new_node, goal) < step_size:
                    if is_path_valid(environment, new_node, goal):
                        tree[goal_tuple] = new_tuple
                        return extract_path(tree, start, goal), explored_nodes

        # Early termination if we've explored a lot
        if i % 100 == 0 and len(nodes) > 500:
            # Try connecting to goal from all nodes
            closest_node = min(nodes, key=lambda n: euclidean_distance(n, goal))
            if euclidean_distance(closest_node, goal) < step_size * 2:
                if is_path_valid(environment, closest_node, goal):
                    tree[goal_tuple] = tuple(closest_node)
                    return extract_path(tree, start, goal), explored_nodes

    return None, explored_nodes  # No path found within iteration limit

def potential_field(environment, start, goal, max_iterations=1000,
                   attractive_coef=1.0, repulsive_coef=100.0, step_size=0.1):
    """Potential Field Method"""
    current = start.copy()
    path = [current.copy()]
    explored_nodes = []

    goal = np.array(goal)

    for _ in range(max_iterations):
        if euclidean_distance(current, goal) < step_size:
            return path, explored_nodes  # Goal reached

        # Compute attractive force
        attractive_force = attractive_coef * (goal - current)

        # Compute repulsive force
        repulsive_force = np.zeros(2)
        for obs in environment.obstacles:
            if isinstance(obs, Obstacle):
                # Get closest point on rectangle
                closest_x = max(obs.x, min(current[0], obs.x + obs.width))
                closest_y = max(obs.y, min(current[1], obs.y + obs.height))
                closest = np.array([closest_x, closest_y])
            elif isinstance(obs, CircleObstacle):
                # Get closest point on circle
                direction = current - np.array([obs.x, obs.y])
                distance = np.linalg.norm(direction)
                if distance < 0.01:  # Very close to center
                    direction = np.array([1, 0])  # Arbitrary direction
                else:
                    direction = direction / distance
                closest = np.array([obs.x, obs.y]) + direction * obs.radius

            # Calculate repulsive force
            to_robot = current - closest
            dist = np.linalg.norm(to_robot)

            if dist == 0:  # Avoid division by zero
                continue

            # Only apply repulsion within a certain range
            influence_range = 3.0
            if dist < influence_range:
                repulsive_force += repulsive_coef * (1.0/dist - 1.0/influence_range) * \
                                  (1.0/(dist**2)) * (to_robot/dist)

        # Calculate total force and normalize
        total_force = attractive_force + repulsive_force
        force_magnitude = np.linalg.norm(total_force)

        if force_magnitude > 0:
            # Move in the direction of the force
            movement = step_size * total_force / force_magnitude
            new_position = current + movement

            # Check if new position is valid
            if environment.is_valid_position(new_position):
                current = new_position
                path.append(current.copy())
                explored_nodes.append(tuple(current))
            else:
                # Add some random movement to escape local minima
                random_direction = environment.random_valid_position() - current
                random_magnitude = np.linalg.norm(random_direction)
                if random_magnitude > 0:
                    random_movement = step_size * random_direction / random_magnitude
                    new_position = current + random_movement
                    if environment.is_valid_position(new_position):
                        current = new_position
                        path.append(current.copy())
                        explored_nodes.append(tuple(current))
        else:
            # Local minimum, add some random movement
            random_direction = environment.random_valid_position() - current
            random_magnitude = np.linalg.norm(random_direction)
            if random_magnitude > 0:
                random_movement = step_size * random_direction / random_magnitude
                new_position = current + random_movement
                if environment.is_valid_position(new_position):
                    current = new_position
                    path.append(current.copy())
                    explored_nodes.append(tuple(current))

    return path, explored_nodes  # Return path even if goal not reached

def create_random_environment():
    """Create an environment with random obstacles"""
    env = Environment(COLS, ROWS)

    # Add some random rectangular obstacles
    for _ in range(10):
        x = np.random.uniform(2, COLS - 5)
        y = np.random.uniform(2, ROWS - 5)
        width = np.random.uniform(1, 3)
        height = np.random.uniform(1, 3)
        env.add_obstacle(Obstacle(x, y, width, height))

    # Add some random circular obstacles
    for _ in range(5):
        x = np.random.uniform(2, COLS - 2)
        y = np.random.uniform(2, ROWS - 2)
        radius = np.random.uniform(0.5, 1.5)
        env.add_obstacle(CircleObstacle(x, y, radius))

    # Set random start and goal positions
    start = env.random_valid_position()
    env.set_start(start[0], start[1])

    goal = env.random_valid_position()
    # Make sure start and goal are far enough apart
    while euclidean_distance(start, goal) < min(COLS, ROWS) / 2:
        goal = env.random_valid_position()
    env.set_goal(goal[0], goal[1])

    return env

def create_maze_environment():
    """Create an environment with a maze-like structure"""
    env = Environment(COLS, ROWS)

    # Create maze walls
    # Vertical walls
    for i in range(2, ROWS-2, 4):
        height = min(ROWS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(5, i, 0.5, height))

    for i in range(4, ROWS-2, 4):
        height = min(ROWS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(10, i, 0.5, height))

    for i in range(2, ROWS-2, 4):
        height = min(ROWS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(15, i, 0.5, height))

    for i in range(4, ROWS-2, 4):
        height = min(ROWS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(20, i, 0.5, height))

    # Horizontal walls
    for i in range(2, COLS-2, 4):
        width = min(COLS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(i, 5, width, 0.5))

    for i in range(4, COLS-2, 4):
        width = min(COLS - i - 1, np.random.randint(2, 4))
        env.add_obstacle(Obstacle(i, 10, width, 0.5))

    # Set start and goal positions
    env.set_start(1, 1)
    env.set_goal(COLS-2, ROWS-2)

    return env

def draw_path(screen, path, color=BLUE, width=2):
    """Draw a path on the screen"""
    if path is None or len(path) < 2:
        return

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        pygame.draw.line(screen, color,
                        (int(p1[0] * GRID_SIZE), int(p1[1] * GRID_SIZE)),
                        (int(p2[0] * GRID_SIZE), int(p2[1] * GRID_SIZE)),
                        width)

def draw_explored_nodes(screen, nodes, color=CYAN, radius=2):
    """Draw explored nodes"""
    for node in nodes:
        pygame.draw.circle(screen, color,
                          (int(node[0] * GRID_SIZE), int(node[1] * GRID_SIZE)),
                          radius)

def calculate_path_length(path):
    """Calculate the total length of a path"""
    if path is None or len(path) < 2:
        return float('inf')

    length = 0
    for i in range(len(path) - 1):
        length += euclidean_distance(path[i], path[i+1])
    return length

def compare_algorithms(environment, algorithm_results):
    """Compare algorithm performance"""
    # Calculate metrics
    for algo_name, result in algorithm_results.items():
        path, explored = result
        if path:
            result['path_length'] = calculate_path_length(path)
            result['explored_count'] = len(explored)
            result['successful'] = True
        else:
            result['path_length'] = float('inf')
            result['explored_count'] = len(explored)
            result['successful'] = False

    # Print results
    print("\n---- Algorithm Comparison ----")
    for algo_name, result in algorithm_results.items():
        path, explored = result
        print(f"{algo_name}:")
        print(f"  Successful: {result['successful']}")
        print(f"  Path Length: {result['path_length']:.2f}")
        print(f"  Explored Nodes: {result['explored_count']}")
    print("-----------------------------")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Path Planning Simulation")
    clock = pygame.time.Clock()

    # Font for rendering text
    font = pygame.font.SysFont('Arial', 12)

    # Create environment - user can choose type
    print("Select environment type:")
    print("1. Random Environment")
    print("2. Maze Environment")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == '2':
        environment = create_maze_environment()
    else:
        environment = create_random_environment()

    # Select algorithm
    print("\nSelect pathfinding algorithm:")
    print("1. A* Algorithm")
    print("2. Dijkstra's Algorithm")
    print("3. RRT Algorithm")
    print("4. Potential Field Method")
    print("5. Compare All Algorithms")

    algo_choice = input("Enter your choice (1-5): ").strip()

    # Run algorithms based on choice
    path = None
    explored_nodes = []
    algorithm_name = ""

    start_time = time.time()

    if algo_choice == '5':
        # Run all algorithms and compare
        algorithm_results = {}

        print("Running A*...")
        path_astar, explored_astar = astar(environment, environment.start, environment.goal)
        algorithm_results['A*'] = (path_astar, explored_astar)

        print("Running Dijkstra's...")
        path_dijkstra, explored_dijkstra = dijkstra(environment, environment.start, environment.goal)
        algorithm_results['Dijkstra'] = (path_dijkstra, explored_dijkstra)

        print("Running RRT...")
        path_rrt, explored_rrt = rrt(environment, environment.start, environment.goal)
        algorithm_results['RRT'] = (path_rrt, explored_rrt)

        print("Running Potential Field...")
        path_pf, explored_pf = potential_field(environment, environment.start, environment.goal)
        algorithm_results['Potential Field'] = (path_pf, explored_pf)

        compare_algorithms(environment, algorithm_results)

        # Set path to visualize (default to A*)
        path = path_astar
        explored_nodes = explored_astar
        algorithm_name = "A*"

    else:
        if algo_choice == '1':
            print("Running A*...")
            path, explored_nodes = astar(environment, environment.start, environment.goal)
            algorithm_name = "A*"
        elif algo_choice == '2':
            print("Running Dijkstra's...")
            path, explored_nodes = dijkstra(environment, environment.start, environment.goal)
            algorithm_name = "Dijkstra"
        elif algo_choice == '3':
            print("Running RRT...")
            path, explored_nodes = rrt(environment, environment.start, environment.goal)
            algorithm_name = "RRT"
        elif algo_choice == '4':
            print("Running Potential Field...")
            path, explored_nodes = potential_field(environment, environment.start, environment.goal)
            algorithm_name = "Potential Field"

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nAlgorithm: {algorithm_name}")
    print(f"Execution Time: {execution_time:.4f} seconds")

    if path:
        print(f"Path Found! Length: {calculate_path_length(path):.2f}")
        print(f"Explored {len(explored_nodes)} nodes")
    else:
        print("No path found!")

    # Main game loop
    running = True
    show_explored = True  # Toggle for showing explored nodes

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    show_explored = not show_explored  # Toggle explored nodes
                elif event.key == pygame.K_s:
                    # Save screenshot
                    pygame.image.save(screen, f"path_planning_{algorithm_name}.png")
                    print("Screenshot saved!")
                elif event.key == pygame.K_q:
                    running = False

        # Draw environment
        environment.draw(screen)

        # Draw explored nodes if toggled on
        if show_explored:
            draw_explored_nodes(screen, explored_nodes)

        # Draw path
        if path:
            draw_path(screen, path)

        # Draw info text
        info_text = [
            f"Algorithm: {algorithm_name}",
            f"Execution Time: {execution_time:.4f} s",
            f"Path Length: {calculate_path_length(path):.2f}" if path else "No path found",
            f"Explored Nodes: {len(explored_nodes)}",
            "Press E to toggle explored nodes",
            "Press S to save screenshot",
            "Press Q to quit"
        ]

        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (5, 5 + i * 15))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
