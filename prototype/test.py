# Packages
import pygame
import sys
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height), pygame.SRCALPHA)
pygame.display.set_caption('Hello World in Pygame')

# Set up the circle
circle_color = (255, 255, 255)
circle_radius = 5
circle_x, circle_y = screen_width // 2, screen_height // 2
circle_speed = 800 # pixels per second

# Set up the trail
trail = []

# Set up the clock
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    # Get the time delta
    dt = clock.tick(60) / 1000  # 60 frames per second

    # Get keys
    keys = pygame.key.get_pressed()

    # Quit game
    if keys[pygame.K_ESCAPE] or pygame.event.get(pygame.QUIT):
        running = False
    
    # Move circle
    move_x, move_y = 0, 0
    if keys[pygame.K_LEFT]:
        move_x -= circle_speed * dt
    if keys[pygame.K_RIGHT]:
        move_x += circle_speed * dt
    if keys[pygame.K_UP]:
        move_y -= circle_speed * dt
    if keys[pygame.K_DOWN]:
        move_y += circle_speed * dt
    if keys[pygame.K_ESCAPE]:
        running = False
    if move_x != 0 and move_y != 0:
        move_x /= math.sqrt(2)
        move_y /= math.sqrt(2)

    # Update circle position
    circle_x += move_x
    circle_y += move_y

    # Wrap around screen
    if circle_x < 0:
        circle_x = screen_width
    elif circle_x > screen_width:
        circle_x = 0
    if circle_y < 0:
        circle_y = screen_height
    elif circle_y > screen_height:
        circle_y = 0

    # Add current position to trail
    trail.append((circle_x, circle_y))
    if len(trail) > 100:  # Limit trail length
        trail.pop(0)

    # Fill the screen with blue
    screen.fill((0, 0, 255))

    # Draw the trail
    for pos, alpha in zip(trail, np.linspace(0, 255, len(trail)).astype(int)):
        trail_surface = pygame.Surface((circle_radius * 2, circle_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(trail_surface, (255, 255, 255, alpha), (circle_radius, circle_radius), circle_radius)
        screen.blit(trail_surface, (pos[0] - circle_radius, pos[1] - circle_radius))

    # Draw the circle
    pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()