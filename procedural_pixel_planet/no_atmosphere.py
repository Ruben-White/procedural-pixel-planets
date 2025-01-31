# %%
# Packages
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

# %%
# Settings
pixels = 100.0
rotation = 0.0
light_origin = (0.25, 0.25)
time_speed = 0.4
dither_size = 2.0
light_border_1 = 0.615
light_border_2 = 0.729
colors = [(0.639216, 0.654902, 0.760784, 1), (0.298039, 0.407843, 0.521569, 1), (0.227451, 0.247059, 0.368627, 1)]
size = 8.0
OCTAVES = 4
seed = 1.012
time = 0.0
should_dither = True

# %%
# Random number generator
def rand(coord):
    coord = np.mod(coord, np.array([1.0, 1.0]) * np.round(size))
    return np.mod(np.sin(np.dot(coord, np.array([12.9898, 78.233]))) * 15.5453 * seed, 1)

# Perlin Noise
def noise(coord):
    i = np.floor(coord)
    f = np.mod(coord, 1)
    a = rand(i)
    b = rand(i + np.array([1.0, 0.0]))
    c = rand(i + np.array([0.0, 1.0]))
    d = rand(i + np.array([1.0, 1.0]))
    cubic = f * f * (3.0 - 2.0 * f)

    return a + (b - a) * cubic[0] + (c - a) * cubic[1] * (1.0 - cubic[0]) + (d - b) * cubic[0] * cubic[1]

# Fractal Brownian Motion
def fbm(coord):
    value = 0.0
    scale = 0.5
    for i in range(OCTAVES):
        value += noise(coord) * scale
        coord *= 2.0
        scale *= 0.5
    return value

# Dithering
def dither(uv1, uv2):
    return np.mod(uv1[0] + uv2[1], 2.0 / pixels) <= 1.0 / pixels

# Rotation
def rotate(coord, angle):
    coord -= 0.5
    coord = np.dot(coord, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    return coord + 0.5

def fragment(UV):
    uv = np.floor(UV * pixels) / pixels
    d_circle = np.linalg.norm(uv - np.array([0.5, 0.5]))
    d_light = np.linalg.norm(uv - np.array(light_origin))
    a = np.where(d_circle < 0.5, 1.0, 0.0)
    dith = dither(uv, UV)
    uv = rotate(uv, rotation)
    fbm1 = fbm(uv)
    d_light += fbm(uv * size + fbm1 + np.array([time * time_speed, 0.0])) * 0.3
    dither_border = 1.0 / pixels * dither_size
    col = np.array(colors[0])
    if d_light > light_border_1:
        col = np.array(colors[1])
        if d_light < light_border_1 + dither_border and (dith or not should_dither):
            col = np.array(colors[0])
    if d_light > light_border_2:
        col = np.array(colors[2])
        if d_light < light_border_2 + dither_border and (dith or not should_dither):
            col = np.array(colors[1])
    return col * a

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros(X.shape + (4,))
for i, j in np.ndindex(X.shape):
    Z[i, j] = fragment(np.array([X[i, j], Y[i, j]]))

axs[0].imshow(Z[:, :, 0])
axs[1].imshow(Z[:, :, 1])
axs[2].imshow(Z[:, :, 2])
axs[3].imshow(Z[:, :, :3])

# %%
# Pygame
pg.init()

# Settings
screen_size = 100

# Screen
screen = pg.display.set_mode((screen_size, screen_size))

# Main loop
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # Time
    time += 1

    # Draw background
    screen.fill((0, 0, 0))

    # Draw
    xs = np.linspace(0, 1, screen_size)
    ys = np.linspace(0, 1, screen_size)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape + (4,))
    for i, j in np.ndindex(X.shape):
        Z[i, j] = fragment(np.array([X[i, j], Y[i, j]]))
    pg.surfarray.blit_array(screen, (Z[:,:,:3] * 255).astype(np.uint8))

    # Update
    pg.display.flip()

pg.quit()

# %%
