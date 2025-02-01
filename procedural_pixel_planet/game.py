# %%
# Packages
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
# Screen settings
screen_size = pg.Vector2(700, 700)

# Planet settings
position = pg.Vector2(screen_size.x / 2, screen_size.y / 2)
radius = 250
pixels = 250//2
angle = 0
front_vec = pg.Vector3(0, 0, 1)

# Light settings
light_vec = pg.Vector3(0, 0, 1)
dithering = True

# Mouse settings
mouse_radius = radius

# Planet class
class Planet:
    # Function to initialise the planet
    def __init__(self, position, radius, pixels,
                 pitch=0, yaw=0, roll=0,
                 light_vec=pg.Vector3(0, 0, 1)):
        
        # Set the properties
        self.position = position
        self.radius = radius
        self.shape = (pixels, pixels)
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        # Get the x and y matrices
        xs = np.linspace(-radius, radius, self.shape[0])
        ys = np.linspace(-radius, radius, self.shape[1])        
        xG, yG = np.meshgrid(xs, ys)

        # Get the mask matrix
        maskG = xG**2 + yG**2 <= radius**2
        
        # Get the z matrix
        zG = np.sqrt(np.where(maskG, radius**2 - xG**2 - yG**2, 0))

        # Get the dithering matrix
        dithers = np.zeros((len(xs), len(ys)), dtype=bool)
        for i in range(len(xs)):
            j = i % 2
            dithers[i, j::2] = True
        ditherG = dithers.reshape(len(xs), len(ys))

        # Get the normal matrix (unit vectors)
        normalG = np.dstack((xG, yG, zG))
        normalG = normalG / np.linalg.norm(normalG, axis=2, keepdims=True)

        # Set the matrices
        self.xG = xG
        self.yG = yG
        self.zG = zG
        self.xrotG = xG.copy()
        self.yrotG = yG.copy()
        self.zrotG = zG.copy()
        self.maskG = maskG
        self.ditherG = ditherG
        self.normalG = normalG

        # Angle the planet
        self.angle_planet(pitch, yaw, roll)

        # Colour the planet
        self.colour_planet()

        # Light the planet
        self.light_planet(light_vec, dithering)

    # Function to angle the planet
    def angle_planet(self, pitch=None, yaw=None, roll=None):
        # Get the pitch, yaw and roll
        if pitch is None:
            pitch = self.pitch
        if yaw is None:
            yaw = self.yaw
        if roll is None:
            roll = self.roll

        # Get the pitch matrix (z-plane)
        pitch_matrix = np.array([[np.cos(pitch), -np.sin(pitch), 0],
                                [np.sin(pitch), np.cos(pitch), 0],
                                [0, 0, 1]])
        
        # Get the yaw matrix (y-plane)
        yaw_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                                [0, 1, 0],
                                [-np.sin(yaw), 0, np.cos(yaw)]])

        # Get the roll matrix (x-plane)
        roll_matrix = np.array([[1, 0, 0],
                                [0, np.cos(roll), -np.sin(roll)],
                                [0, np.sin(roll), np.cos(roll)]])

        # Combine pitch, yaw and roll matrices
        combined_matrix = np.dot(np.dot(pitch_matrix, yaw_matrix), roll_matrix)
        
        # Rotate xG, yG and zG
        xrotG, yrotG, zrotG = np.dot(np.dstack((self.yG, self.xG, self.zG)), combined_matrix).T

        # Set the matrice
        self.xrotG = xrotG
        self.yrotG = yrotG
        self.zrotG = zrotG

        # Set the pitch, yaw and roll
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        
    # Function to colour the planet
    def colour_planet(self):
        # Get colour matrix (0-255)
        colour_type = 'rotated_quadrants'
        if colour_type == 'red':
            colourG = np.zeros(self.shape + (4,))
            colourG[:, :, 0] = 255
        elif colour_type == 'normal':
            colourG - np.zeros(self.shape + (4,))
            colourG[:, :, 0] = np.interp(self.normalG[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(self.normalG[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(self.normalG[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'quadrants':
            colourG = np.zeros(self.shape + (4,))
            colourG[:, :, 0] = np.where(self.xG > 0, 255, 0)
            colourG[:, :, 1] = np.where(self.yG > 0, 255, 0)
            colourG[:, :, 2] = np.where(self.zG > 0, 255, 0)
        elif colour_type == 'rotated_normal':
            colourG = np.zeros(self.shape + (4,))
            colourG[:, :, 0] = np.interp(self.xrotG, (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(self.yrotG, (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(self.zrotG, (-1, 1), (0, 255))
        elif colour_type == 'rotated_quadrants':
            colourG = np.zeros(self.shape + (4,))
            colourG[:, :, 0] = np.where(self.xrotG > 0, 255, 0)
            colourG[:, :, 1] = np.where(self.yrotG > 0, 255, 0)
            colourG[:, :, 2] = np.where(self.zrotG > 0, 255, 0)
        
        # Set the alpha channel of the colour matrix
        colourG[:, :, 3] = np.where(self.maskG, 255, 0)

        # Set the matrices
        self.colourG = colourG
    
    # Function to light the planet
    def light_planet(self, light_vec, dithering=False):
        # Normalize the light vector
        light_vec = light_vec.normalize()

        # Calculate the light intensity
        intensityG = np.dot(self.normalG, light_vec)

        # Normalize the intensity
        intensityG = (intensityG + 1) / 2
        #intensityG = np.clip(intensityG, 0, 1)

        # Apply dithering
        if dithering:
            # Get the linemask
            levels = 8
            thres = 0.05
            diffG = np.abs(intensityG - np.round(intensityG * levels) / levels)
            linemaskG = diffG > thres #np.percentile(diffG, percentile)

            # Get the rounded and ceiled intensities
            round_intensityG = np.round(intensityG * levels) / levels
            ceil_intensityG = np.ceil(intensityG * levels) / levels

            # Apply dithering
            intensityG = np.where(np.logical_and(self.ditherG, linemaskG), ceil_intensityG, round_intensityG)
        else:
            # Round the intensity
            levels = 8
            intensityG = np.round(intensityG * levels) / levels

        # Set the light matrix
        self.intensityG = intensityG

    # Function to draw the planet
    def draw_planet(self, screen):
        # Multiply the colour by the light intensity
        colourG = self.colourG * np.expand_dims(self.intensityG, axis=2)
        colourG[:, :, 3] = self.colourG[:, :, 3]

        # Draw the planet on its surface
        surface = pg.image.frombuffer(colourG.astype(np.uint8).tobytes(), (self.shape[0], self.shape[1]), 'RGBA')

        # Scale surface to pixel size
        surface = pg.transform.scale(surface, (2 * self.radius, 2 * self.radius))
        # Blit the planet surface onto the main screen
        screen.blit(surface, (self.position.x - self.radius, self.position.y - self.radius))
        
# Function to get mouse vector
def get_mouse_vec(mouse_radius, screen_size):
    # Get the mouse position
    mouse_pos = pg.Vector2(pg.mouse.get_pos())

    # Get the mouse vector
    mouse_vec = pg.Vector3(mouse_pos.x - screen_size.x / 2, mouse_pos.y - screen_size.y / 2, 0)
    if mouse_radius**2 > mouse_vec.x**2 + mouse_vec.y**2:
        mouse_vec.z = np.sqrt(mouse_radius**2 - mouse_vec.x**2 - mouse_vec.y**2)
    else:
        mouse_vec.z = -np.sqrt(mouse_vec.x**2 + mouse_vec.y**2 - mouse_radius**2)

    # Normalize the mouse vector
    mouse_vec = mouse_vec.normalize()

    # Return the mouse vector
    return mouse_vec

# Initialise the planet
planet = Planet(position, radius, pixels,
                pitch=0, yaw=0, roll=0,
                light_vec=front_vec)

# Plot the planet
plot = True
if plot:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colourG = planet.colourG * np.expand_dims(planet.intensityG, axis=2)
    colourG[:, :, 3] = planet.colourG[:, :, 3]
    colourG = colourG.astype(np.uint8)
    ax.imshow(colourG)
    ax.set_aspect('equal')
    ax.set_facecolor('grey')

    fig, axs= plt.subplots(2, 3, figsize=(12, 4))
    axs = axs.flatten()
    for ax, _G in zip(axs, [planet.xG, planet.yG, planet.zG,
                            planet.xrotG, planet.yrotG, planet.zrotG]):
        p = ax.imshow(_G)
        fig.colorbar(p, ax=ax)
        ax.set_aspect('equal')
        ax.set_facecolor('grey')
    fig.tight_layout()

# %%
# Intialise pygame
pg.init()

# Initialise the screen
screen = pg.display.set_mode(screen_size)

# Initialise the clock
clock = pg.time.Clock()

# Font settings
font = pg.font.Font('slkscre.ttf', 32)

# Initialise the pressed variables
pressed_d = False
pressed_m = False
mouse_control = 'light'

# Game loop
running = True
while running:
    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # Toggle dithering
    keys = pg.key.get_pressed()
    if keys[pg.K_d] and not pressed_d:
        dithering = not dithering
        pressed_d = True
    elif not keys[pg.K_d]:
        pressed_d = False

    # Pitch the planet (numpad 7 and 9)
    if keys[pg.K_KP9]:
        planet.angle_planet(pitch=planet.pitch + 0.05)
    if keys[pg.K_KP7]:
        planet.angle_planet(pitch=planet.pitch - 0.05)

    # Yaw the planet (numpad 4 and 6)
    if keys[pg.K_KP6]:
        planet.angle_planet(yaw=planet.yaw + 0.05)
    if keys[pg.K_KP4]:
        planet.angle_planet(yaw=planet.yaw - 0.05)

    # Roll the planet (numapd 8 and 5)
    if keys[pg.K_KP8]:
        planet.angle_planet(roll=planet.roll + 0.05)
    if keys[pg.K_KP5]:
        planet.angle_planet(roll=planet.roll - 0.05)

    # Colour the planet
    planet.colour_planet()

    # Light the planet
    light_vec = get_mouse_vec(mouse_radius, screen_size)
    planet.light_planet(light_vec, dithering)
    
    # Quit the game by pressing 'esc'
    if keys[pg.K_ESCAPE]:
        running = False

    # Clear the screen
    screen.fill(pg.Color('black'))

    # Draw the planet
    planet.draw_planet(screen)

    # Draw the framerate counter
    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, pg.Color('white'))
    screen.blit(fps_text, (10, 10))
    
    # Update the display
    pg.display.flip()

    # Tick the clock
    clock.tick(60)

pg.quit()

# %%
