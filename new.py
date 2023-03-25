import pygame
import math

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Define the size of the screen
SCREEN_WIDTH = 5070/10 + 100
SCREEN_HEIGHT = 2600/10 + 100

# Initialize Pygame
pygame.init()


class IcebergGame:
    def __init__(self):
        # Set the title of the window
        pygame.display.set_caption("Iceberg Game")

        # Create the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Create a font
        self.font = pygame.font.SysFont(None, 30)

        # Define the size and duration of the attack animation
        self.attack_size = 10
        self.attack_step_size = 20

        # Define the list of icebergs
        self.icebergs = [
            {'x': 1300/10,  'y': 600/10, "troops": 11},
            {'x': 1880/10,  'y': 1800/10, "troops": 11},
            {'x': 2433/10, 'y': 1200/10, "troops": 10},
            {'x': 2433/10, 'y': 2600/10, "troops": 10},
            {'x': 3967/10, 'y': 1200/10, "troops": 10},
            {'x': 3967/10, 'y': 2600/10, "troops": 10},
            {'x': 4520/10, 'y': 1800/10, "troops": 10},
            {'x': 5070/10, 'y': 600/10, "troops": 11},
        ]

        # Define the list of attacks
        self.attacks = []

    def send_attack(self, start_loc, end_loc, num_steps):
        # Convert start_loc and end_loc to integers
        start_loc = [int(coord) for coord in start_loc]
        end_loc = [int(coord) for coord in end_loc]

        # Find the start and end icebergs
        start = None
        end = None
        for iceberg in self.icebergs:
            if iceberg['x'] == start_loc[0] and iceberg['y'] == start_loc[1]:
                start = iceberg
            elif iceberg['x'] == end_loc[0] and iceberg['y'] == end_loc[1]:
                end = iceberg
        if not start or not end:
            print("Invalid start or end location.")
            return

        # Add the attack to the list
        self.attacks.append({'start': start, 'end': end, 'num_steps': num_steps, 'size': self.attack_size, 'step': 0})

        # Start the attack animation
        self.animate_attack(start, end, num_steps)

    def animate_attack(self, start, end, num_steps):
        # Calculate the distance and direction between the icebergs
        distance = math.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
        direction_x = (end['x'] - start['x']) / distance
        direction_y = (end['y'] - start['y']) / distance

        # Calculate the size of each
        step_size = distance / num_steps

        # Calculate the position of the attack for this step
        attack_x = int(start['x'] + direction_x * step_size * self.attacks[-1]['step'])
        attack_y = int(start['y'] + direction_y * step_size * self.attacks[-1]['step'])

        # Draw the attack
        pygame.draw.circle(self.screen, RED, (attack_x, attack_y), self.attack_size)

        # Increment the step counter
        self.attacks[-1]['step'] += 1

        # Check if the attack is finished
        if self.attacks[-1]['step'] > num_steps:
            # Remove the attack from the list
            self.attacks.pop()
        else:
            # Add the attack to the front of the list
            self.attacks.insert(0, {'start': start, 'end': end, 'num_steps': num_steps, 'size': self.attack_size, 'step': 0})

    def step(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear the screen
        self.screen.fill(WHITE)

        # Draw the icebergs
        for iceberg in self.icebergs:
            pygame.draw.circle(self.screen, BLUE, (iceberg['x'], iceberg['y']), 30)
            text = self.font.render(str(iceberg["troops"]), True, BLACK)
            self.screen.blit(text, (iceberg['x'] - 10, iceberg['y'] - 10))

        # Draw the attacks
        for attack in self.attacks:
            start = attack['start']
            end = attack['end']
            num_steps = attack['num_steps']
            size = attack['size']
            step = attack['step']
            self.animate_attack(start, end, num_steps)

        # Update the screen
        pygame.display.flip()