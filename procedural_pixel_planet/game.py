# %%
# Packages
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import os

# NOTE: ORIENTATION AND ROTATION USE DIFFERENT AXIS SYSTEMS

# %%
# Planet
class Planet:
    # Initialise planet
    def __init__(self, radius, pixels,
                 position=pg.Vector2(0, 0),
                 pitch=0, yaw=0, roll=0, # TODO: REMOVE AND REPLACE WITH ORIENTATION
                 orientation=pg.Vector3(0, 1, 0),
                 rotation_speed=pg.Vector3(0.1, 0, 0),
                 time=0, light=pg.Vector3(0, 0, 1), dithering=True):
        
        # Set planet properties
        self.radius = radius
        self.shape = (pixels, pixels)
        self.position = position
        self.pitch = pitch #TODO: REMOVE
        self.yaw = yaw # TODO: REMOVE
        self.roll = roll # TODO: REMOVE
        self.orientation = orientation.normalize()
        self.rotation_speed = rotation_speed
        self.rotation = rotation_speed * time
        
        # Get x and y matrices
        xs = np.linspace(-radius, radius, self.shape[0])
        ys = np.linspace(-radius, radius, self.shape[1])        
        xG, yG = np.meshgrid(xs, ys)

        # Get mask matrix
        maskG = xG**2 + yG**2 <= radius**2
        
        # Get z matrix
        zG = np.sqrt(np.where(maskG, radius**2 - xG**2 - yG**2, 0))

        # Get the normal matrix (unit vectors)
        normalG = np.dstack((xG, yG, zG))
        normalG = normalG / np.linalg.norm(normalG, axis=2, keepdims=True)

        # Get the dithering matrix
        dithers = np.zeros((len(xs), len(ys)), dtype=bool)
        for i in range(len(xs)):
            j = i % 2
            dithers[i, j::2] = True
        ditherG = dithers.reshape(len(xs), len(ys))

        # Set matrices
        self.xG = xG
        self.yG = yG
        self.zG = zG
        self.maskG = maskG
        self.normalG = normalG
        self.ditherG = ditherG

        # Reorient planet
        self.reorient(pitch=pitch, yaw=yaw, roll=roll)

        # Rotate planet
        self.rotate(time=time, rotation_speed=rotation_speed)
        
        # Colour planet
        self.colour()

        # Illuminate planet
        self.illuminate(light=light, dithering=dithering)

    # Reorient planet
    # TODO: should take orientation vector in future
    # NOTE: DO NOT FORGET TO NORMALISE ORIENTATION VECTOR
    def reorient(self, pitch=None, yaw=None, roll=None):
        # Get pitch, yaw and roll
        if pitch is None:
            pitch = self.pitch
        if yaw is None:
            yaw = self.yaw
        if roll is None:
            roll = self.roll

        # Get pitch, yaw and roll matrices
        pitch_matrix = np.array([[1, 0, 0],
                                 [0, np.cos(pitch), -np.sin(pitch)],
                                 [0, np.sin(pitch), np.cos(pitch)]])
        
        yaw_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                               [0, 1, 0],
                               [-np.sin(yaw), 0, np.cos(yaw)]])
        
        roll_matrix = np.array([[np.cos(roll), -np.sin(roll), 0],
                                [np.sin(roll), np.cos(roll), 0],
                                [0, 0, 1]])

        # Combine pitch, yaw and roll matrices
        combined_matrix = np.dot(np.dot(pitch_matrix, yaw_matrix), roll_matrix)
        
        # Reorient xG, yG and zG matrices
        xG2, yG2, zG2 = np.dot(np.dstack((self.yG, self.xG, self.zG)), combined_matrix).T

        # Set reorineted matrices
        self.xG2 = xG2
        self.yG2 = yG2
        self.zG2 = zG2

        # Set the pitch, yaw and roll
        self.pitch = pitch # TODO: REPLACE
        self.yaw = yaw # TODO: REPLACE
        self.roll = roll # TODO: REPLACE

    # Rotate planet
    def rotate(self, time, rotation_speed=None):
        # Get time and rotation speed
        if rotation_speed is None:
            rotation_speed = self.rotation_speed

        # Get rotation
        rotation = rotation_speed * time
        
        # If rotation is zero, return oriented axes
        if rotation.length() == 0:
            self.xG3 = self.xG2
            self.yG3 = self.yG2
            self.zG3 = self.zG2
            return
        
        # Get rotation matrix
        rotate_matrix = np.array([[np.cos(rotation.y) * np.cos(rotation.z),
                                   np.cos(rotation.y) * np.sin(rotation.z),
                                   -np.sin(rotation.y)],
                                  [np.sin(rotation.x) * np.sin(rotation.y) * np.cos(rotation.z) - np.cos(rotation.x) * np.sin(rotation.z),
                                   np.sin(rotation.x) * np.sin(rotation.y) * np.sin(rotation.z) + np.cos(rotation.x) * np.cos(rotation.z),
                                   np.sin(rotation.x) * np.cos(rotation.y)],
                                  [np.cos(rotation.x) * np.sin(rotation.y) * np.cos(rotation.z) + np.sin(rotation.x) * np.sin(rotation.z),
                                   np.cos(rotation.x) * np.sin(rotation.y) * np.sin(rotation.z) - np.sin(rotation.x) * np.cos(rotation.z),
                                   np.cos(rotation.x) * np.cos(rotation.y)]])
        
        # Rotate xG, yG and zG matrices
        xG3, yG3, zG3 = np.dot(np.dstack((self.yG2, self.xG2, self.zG2)), rotate_matrix).T

        # Set reoriented matrices
        self.xG3 = xG3
        self.yG3 = yG3
        self.zG3 = zG3

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
        elif colour_type == 'normal':
            colourG[:, :, 0] = np.interp(self.normalG[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(self.normalG[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(self.normalG[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'normal2':
            normalG2 = np.dstack((self.xG2, self.yG2, self.zG2))
            normalG2 = normalG2 / np.linalg.norm(normalG2, axis=2, keepdims=True)
            colourG[:, :, 0] = np.interp(normalG2[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(normalG2[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(normalG2[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'normal3':
            normalG3 = np.dstack((self.xG3, self.yG3, self.zG3))
            normalG3 = normalG3 / np.linalg.norm(normalG3, axis=2, keepdims=True)
            colourG[:, :, 0] = np.interp(normalG3[:, :, 0], (-1, 1), (0, 255))
            colourG[:, :, 1] = np.interp(normalG3[:, :, 1], (-1, 1), (0, 255))
            colourG[:, :, 2] = np.interp(normalG3[:, :, 2], (-1, 1), (0, 255))
        elif colour_type == 'quadrants':
            colourG[:, :, 0] = np.where(self.xG > 0, 255, 0)
            colourG[:, :, 1] = np.where(self.yG > 0, 255, 0)
            colourG[:, :, 2] = np.where(self.zG > 0, 255, 0)
        elif colour_type == 'quadrants2':
            colourG[:, :, 0] = np.where(self.xG2 > 0, 255, 0)
            colourG[:, :, 1] = np.where(self.yG2 > 0, 255, 0)
            colourG[:, :, 2] = np.where(self.zG2 > 0, 255, 0)
        elif colour_type == 'quadrants3':
            colourG[:, :, 0] = np.where(self.xG3 > 0, 255, 0)
            colourG[:, :, 1] = np.where(self.yG3 > 0, 255, 0)
            colourG[:, :, 2] = np.where(self.zG3 > 0, 255, 0)
        else:
            colourG[:, :, 0] = np.where(self.xG3 > 0, 255*0.5, 0)
            colourG[:, :, 1] = np.where(self.yG3 > 0, 255*0.5, 0)
            colourG[:, :, 2] = np.where(self.zG3 > 0, 255*0.5, 0)
            colourG[:, :, 0] = np.where(self.xG2 > self.radius*0.8, 255, colourG[:, :, 0])
            colourG[:, :, 1] = np.where(self.yG2 > self.radius*0.8, 255, colourG[:, :, 1])
            colourG[:, :, 2] = np.where(self.zG2 > self.radius*0.8, 255, colourG[:, :, 2])
        
        # Set the alpha channel of the colour matrix
        colourG[:, :, 3] = np.where(self.maskG, 255, 0)

        # Set the matrices
        self.colourG = colourG
    
    # Illuminate planet
    def illuminate(self, light, dithering):
        # Normalize light vector
        light = light.normalize()

        # Calculate light intensity
        intensityG = np.dot(self.normalG, light)

        # Normalize light intensity
        intensityG = (intensityG + 1) / 2
        #intensityG = np.clip(intensityG, 0, 1)

        # Apply dithering
        levels = 4   # Number of rings
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
orientation = pg.Vector3(0, 1, 0)
rotation_speed = pg.Vector3(0, 1, 0)

# Light settings
light = pg.Vector3(0, 0, 1)
dithering = True

# Mouse settings
mouse_radius = radius

# %%
# Preview planet
planet = Planet(radius, pixels,
                position=position,
                pitch=0, yaw=0, roll=0, #TODO: REMOVE
                orientation=orientation,
                rotation_speed=rotation_speed,
                time=0, light=light, dithering=dithering)

# Plot planet
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
colourG = planet.colourG * np.expand_dims(planet.intensityG, axis=2)
colourG[:, :, 3] = planet.colourG[:, :, 3]
colourG = colourG.astype(np.uint8)
ax.imshow(colourG)
ax.set_aspect('equal')
ax.set_facecolor('grey')

# Plot axes
fig, axs= plt.subplots(3, 3, figsize=(12, 4))
axs = axs.flatten()
for ax, _G in zip(axs, [planet.xG, planet.yG, planet.zG,
                        planet.xG2, planet.yG2, planet.zG2,
                        planet.xG3, planet.yG3, planet.zG3]):
    p = ax.imshow(_G)
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

    # TODO: REPLACE THIS PART WITH VECTOR
    # Pitch planet
    if keys[pg.K_KP8]:
        planet.reorient(pitch=planet.pitch + 0.05)
    if keys[pg.K_KP5]:
        planet.reorient(pitch=planet.pitch - 0.05)

    # Yaw planet
    if keys[pg.K_KP6]:
        planet.reorient(yaw=planet.yaw + 0.05)
    if keys[pg.K_KP4]:
        planet.reorient(yaw=planet.yaw - 0.05)

    # Roll planet
    if keys[pg.K_KP9]:
        planet.reorient(roll=planet.roll + 0.05)
    if keys[pg.K_KP7]:
        planet.reorient(roll=planet.roll - 0.05)

    # Get time in seconds
    time = pg.time.get_ticks() / 1000

    # Get mouse vector
    mouse = get_mouse_vec(mouse_radius=mouse_radius, screen_size=screen_size)

    # Clear screen
    screen.fill(pg.Color('black'))

    # Draw planet
    planet.draw(screen=screen, time=time, light=mouse, dithering=dithering)

    # Draw framerate counter
    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, pg.Color('white'))
    screen.blit(fps_text, (10, 10))
    
    # Update display
    pg.display.flip()

    # Tick clock
    clock.tick(100)

# Quit game
pg.quit()

# %%
