# %%
# Packages
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO: change position and radius to be dependent on distance in 3D space. Add camera controls


# %%
# Planet
class Planet:
    # Initialise planet
    def __init__(self, radius, pixels,
                 position=pg.Vector3(0, 0, 0),
                 orientation=pg.Vector3(0, 1, 0),
                 rotation_speed=pg.Vector3(0.1, 0, 0),
                 time=0, light=pg.Vector3(0, 0, 1), dithering=True):
        
        # Set planet properties
        self.radius = radius
        self.shape = (pixels, pixels)
        
        self.position = position
        self.orientation = orientation       
        self.rotation_speed = rotation_speed 
        self.rotation = rotation_speed * time
        
        # Get x and y matrices
        xs = np.linspace(-radius, radius, self.shape[0])
        ys = np.linspace(-radius, radius, self.shape[1])        
        xG, yG = np.meshgrid(xs, ys)

        # Get mask matrix
        maskG = xG**2 + yG**2 < radius**2
        
        # Get z matrix
        zG = np.sqrt(np.where(maskG, radius**2 - xG**2 - yG**2, 0))

        # Get normal matrix
        normalG1 = np.dstack((xG, yG, zG))

        # Get the dithering matrix
        dithers = np.zeros((len(xs), len(ys)), dtype=bool)
        for i in range(len(xs)):
            j = i % 2
            dithers[i, j::2] = True
        ditherG = dithers.reshape(len(xs), len(ys))

        # Set matrices
        self.normalG1 = normalG1
        self.maskG = maskG
        self.ditherG = ditherG

        # orient planet
        self.orient(orientation=orientation)

        # Rotate planet
        self.rotate(time=time, rotation_speed=rotation_speed)
        
        # Colour planet
        self.colour()

        # Illuminate planet
        self.illuminate(light=light, dithering=dithering)

    # orient planet
    def orient(self, orientation):
        # Get rotation matrix
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(orientation.x), -np.sin(orientation.x)],
                                      [0, np.sin(orientation.x), np.cos(orientation.x)]])
        rotation_matrix_y = np.array([[np.cos(orientation.y), 0, np.sin(orientation.y)],
                                      [0, 1, 0],
                                      [-np.sin(orientation.y), 0, np.cos(orientation.y)]])
        rotation_matrix_z = np.array([[np.cos(orientation.z), -np.sin(orientation.z), 0],
                                      [np.sin(orientation.z), np.cos(orientation.z), 0],
                                      [0, 0, 1]])
        rotation_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
        
        # Rotate normal matrix
        normalV1 = self.normalG1.reshape(-1, 3).T                      # Get normal matrix as vector
        normalV2 = np.dot(rotation_matrix, normalV1)                   # Rotate normal vector
        normalG2 = normalV2.T.reshape(self.shape[0], self.shape[1], 3) # Reshape normal vector to matrix
        
        # Set oriented matrices
        self.normalG2 = normalG2
        
        # Set orientation
        self.orientation = orientation
    
    # Rotate planet
    def rotate(self, time, rotation_speed=None):
        # Get time and rotation speed
        if rotation_speed is None:
            rotation_speed = self.rotation_speed

        # Get rotation
        rotation = rotation_speed * time
        
        # Get rotation matrix
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(rotation.x), -np.sin(rotation.x)],
                                      [0, np.sin(rotation.x), np.cos(rotation.x)]])
        rotation_matrix_y = np.array([[np.cos(rotation.y), 0, np.sin(rotation.y)],
                                      [0, 1, 0],
                                      [-np.sin(rotation.y), 0, np.cos(rotation.y)]])
        rotation_matrix_z = np.array([[np.cos(rotation.z), -np.sin(rotation.z), 0],
                                      [np.sin(rotation.z), np.cos(rotation.z), 0],
                                      [0, 0, 1]])
        rotate_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
        
        # Rotate xG, yG and zG matrices
        normalV2 = self.normalG2.reshape(-1, 3).T                      # Get normal matrix as vector
        normalV3 = np.dot(rotate_matrix, normalV2)                     # Rotate normal vector
        normalG3 = normalV3.T.reshape(self.shape[0], self.shape[1], 3) # Reshape normal vector to matrix
        
        # Set oriented matrices
        self.normalG3 = normalG3
        
        # Set rotation speed and rotation
        self.rotation_speed = rotation_speed
        self.rotation = rotation
        
    # Colour planet
    def colour(self):
        # Get colour matrix
        colour_type = 'debug'
        colourG = np.zeros(self.shape + (4,))
        if colour_type == 'red':
            colourG[:, :, 0] = 255
        elif colour_type == 'normal1':
            normalG = self.normalG1 / np.linalg.norm(self.normalG1, axis=2, keepdims=True)
            colourG[:, :, 0] = np.interp(normalG[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(normalG[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(normalG[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'normal2':
            normalG = self.normalG2 / np.linalg.norm(self.normalG2, axis=2, keepdims=True)
            colourG[:, :, 0] = np.interp(normalG[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(normalG[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(normalG[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'normal3':
            normalG = self.normalG3 / np.linalg.norm(self.normalG3, axis=2, keepdims=True)
            colourG[:, :, 0] = np.interp(normalG[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(normalG[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(normalG[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'quadrants1':
            normalG = self.normalG1 / np.linalg.norm(self.normalG1, axis=2, keepdims=True)
            colourG[:, :, 0] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 1] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 2] = np.where(normalG > 0, 255, 0)
        elif colour_type == 'quadrants2':
            normalG = self.normalG2 / np.linalg.norm(self.normalG2, axis=2, keepdims=True)
            colourG[:, :, 0] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 1] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 2] = np.where(normalG > 0, 255, 0)
        elif colour_type == 'quadrants3':
            normalG = self.normalG3 / np.linalg.norm(self.normalG3, axis=2, keepdims=True)
            colourG[:, :, 0] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 1] = np.where(normalG > 0, 255, 0)
            colourG[:, :, 2] = np.where(normalG > 0, 255, 0)
        elif colour_type == 'debug':
            normalG2 = self.normalG2 / np.linalg.norm(self.normalG2, axis=2, keepdims=True)
            normalG3 = self.normalG3 / np.linalg.norm(self.normalG3, axis=2, keepdims=True)
            colourG[:, :, 0] = np.where(normalG3[:, :, 0] > 0, 255*0.5, 0)
            colourG[:, :, 1] = np.where(normalG3[:, :, 1] > 0, 255*0.5, 0)
            colourG[:, :, 2] = np.where(normalG3[:, :, 2] > 0, 255*0.5, 0)
            colourG[:, :, 0] = np.where(normalG2[:, :, 0] > 0.8, 255, colourG[:, :, 0])
            colourG[:, :, 1] = np.where(normalG2[:, :, 1] > 0.8, 255, colourG[:, :, 1])
            colourG[:, :, 2] = np.where(normalG2[:, :, 2] > 0.8, 255, colourG[:, :, 2])
        
        # Set the alpha channel of the colour matrix
        colourG[:, :, 3] = np.where(self.maskG, 255, 0)

        # Set the matrices
        self.colourG = colourG
    
    # Illuminate planet
    def illuminate(self, light, dithering):
        # Normalize light vector
        light = light.normalize()

        # Normalize normal vector
        normalG = self.normalG1 / np.linalg.norm(self.normalG1, axis=2, keepdims=True)

        # Calculate light intensity
        intensityG = np.dot(normalG, light)

        # Normalize light intensity
        intensityG = (intensityG + 1) / 2
        #intensityG = np.clip(intensityG, 0, 1)

        # Apply dithering
        levels = 128   # Number of rings
        thres = 0.05 # Width of dither rings
        if dithering:
            # Dither ring settings

            # Get ring maks
            diffG = np.abs(intensityG - np.round(intensityG * levels) / levels)
            ringmaskG = diffG > thres

            # Get rounded and floored light intensity
            round_intensityG = np.round(intensityG * levels) / levels
            floor_intensityG = np.floor(intensityG * levels) / levels

            # Get dithered light intensity
            intensityG = np.where(np.logical_and(self.ditherG, ringmaskG), floor_intensityG, round_intensityG)
        else:
            # Get rounded light intensity
            intensityG = np.round(intensityG * levels) / levels

        # Set light intensity matrix
        self.intensityG = intensityG

    # Draw planet
    # # TODO: OPTIMISE, COLOUR ONLY NEEDS UPDATING IF ORIENTATION ROTATION OR TIME
    # HAS CHANGED. SIMILARLY, ILLUMINATE ONLY NEEDS UPDATING IF LIGHT OR DITHERING
    # HAS CHANGED. STORE TIME, LIGHT AND DITHERING LOCALLY TOO.
    def draw(self, screen, time=0, light=pg.Vector3(0, 0, 1), dithering=True):
        # Rotate planet
        self.rotate(time=time)

        # Colour planet
        self.colour()

        # Illuminate planet
        self.illuminate(light=light, dithering=dithering)

        # Multiply colour by light intensity (for RGB, not for A)
        colourG = self.colourG * np.expand_dims(self.intensityG, axis=2)
        colourG[:, :, 3] = self.colourG[:, :, 3]

        # Draw planet on surface
        surface = pg.image.frombuffer(colourG.astype(np.uint8).tobytes(), (self.shape[0], self.shape[1]), 'RGBA')

        # Scale surface based on radius
        surface = pg.transform.scale(surface, (2 * self.radius, 2 * self.radius))

        # Blit surface face onto screen
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

# %%
# Settings
# Screen settings
screen_size = pg.Vector2(700, 700)

# Planet settings
position = pg.Vector2(screen_size.x / 2, screen_size.y / 2)
radius = 250
pixels = 250//2
orientation = pg.Vector3(0, 0, 0)
rotation_speed = pg.Vector3(0, 2*np.pi/10, 0)

# Light settings
light = pg.Vector3(0, 0, 1)
dithering = True

# Mouse settings
mouse_radius = radius

# %%
# Preview planet
planet = Planet(radius, pixels,
                position=position,
                orientation=orientation,
                rotation_speed=rotation_speed,
                time=0.01, light=light, dithering=dithering)

# Plot planet
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
colourG = planet.colourG * np.expand_dims(planet.intensityG, axis=2)
colourG[:, :, 3] = planet.colourG[:, :, 3]
colourG = colourG.astype(np.uint8)
ax.imshow(colourG)
ax.set_aspect('equal')
ax.set_facecolor('grey')

# Plot axes
fig, axs= plt.subplots(3, 3, figsize=(6, 4))
for axs_, normalG in zip(axs, [planet.normalG1, planet.normalG2, planet.normalG3]):
    for ax, i in zip(axs_, range(3)):
        p = ax.imshow(normalG[:, :, i])
        fig.colorbar(p, ax=ax)
        ax.set_aspect('equal')
        ax.set_facecolor('grey')
fig.tight_layout()

# %%
# Simulate planet
# Intialise pygame
pg.init()

# Initialise the screen
screen = pg.display.set_mode(screen_size)
#screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)

# Initialise the clock
clock = pg.time.Clock()

# Font settings
font = pg.font.Font('slkscre.ttf', 32)

# Initialise the pressed variables
pressed_d = False

# Game loop
running = True
while running:
    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    # Quit game
    keys = pg.key.get_pressed()
    if keys[pg.K_ESCAPE]:
        running = False

    # Toggle dithering
    if keys[pg.K_d] and not pressed_d:
        dithering = not dithering
        pressed_d = True
    elif not keys[pg.K_d]:
        pressed_d = False
    
    # Orient planet
    orientation = planet.orientation
    if keys[pg.K_KP8]:
        orientation.x -= 0.1
    if keys[pg.K_KP5]:
        orientation.x += 0.1
    if keys[pg.K_KP6]:
        orientation.y -= 0.1
    if keys[pg.K_KP4]:
        orientation.y += 0.1
    if keys[pg.K_KP9]:
        orientation.z -= 0.1
    if keys[pg.K_KP7]:
        orientation.z += 0.1
    planet.orient(orientation=orientation)

    # Get time in seconds
    time = pg.time.get_ticks() / 1000

    # Get mouse vector
    mouse = get_mouse_vec(mouse_radius=mouse_radius, screen_size=screen_size)

    # Clear screen
    screen.fill(pg.Color('grey'))

    # Draw planet
    planet.draw(screen=screen, time=time, light=mouse, dithering=dithering)

    # Draw framerate counter
    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, pg.Color('white'))
    screen.blit(fps_text, (10, 10))

    # Draw time counter
    time_text = font.render(f"Time: {time:.2f}", True, pg.Color('white'))
    screen.blit(time_text, (10, 50))
    
    # Update display
    pg.display.flip()

    # Tick clock
    clock.tick(100)

# Quit game
pg.quit()

# %%
