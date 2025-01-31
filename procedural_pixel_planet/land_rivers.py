# %%
# Packages
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
# Settings
pixels = 100
rotation = 0.0
light_origin = (0.39, 0.39)
time_speed = 0.2
dither_size = 2.0
should_dither = True
light_border_1 = 0.4
light_border_2 = 0.5
river_cutoff = 0.1

colors = [(0.639216, 0.654902, 0.760784, 1),
          (0.298039, 0.407843, 0.521569, 1),
          (0.227451, 0.247059, 0.368627, 1),
          (0.639216, 0.654902, 0.760784, 1),
          (0.298039, 0.407843, 0.521569, 1),
          (0.227451, 0.247059, 0.368627, 1)]

size = 5
OCTAVES = 4
seed = 1
time_ = 0.0

# %%
# Function to get random number
def rand(coord):
    # Wrap around coordinates within width (2*size) and height (size)
    coord = np.mod(coord, np.array([2.0, 1.0]) * np.round(size))

    # Psuedo-random number generator
    value = np.mod(np.sin(np.dot(coord, np.array([12.9898, 78.233]))) * 15.5453 * seed, 1)

    # Return random number     
    return value

def rand_arr(xG, yG):
    # Wrap around coordinates within width (2*size) and height (size)
    xG = np.mod(xG, 2.0 * size)
    yG = np.mod(yG, size)

    # Psuedo-random number generator
    valueG = np.mod(np.sin(np.dot(np.dstack((xG, yG)), np.array([12.9898, 78.233]))) * 43758.5453 * seed, 1)
    
    # Return random numbers
    return valueG

# Function to get noise
def noise(coord):
    # Get integer and fractional part of coordinate
    i = np.floor(coord)
    f = np.mod(coord, 1)

    # Get random numbers for 4 corners of cell
    a = rand(i)
    b = rand(i + np.array([1.0, 0.0]))
    c = rand(i + np.array([0.0, 1.0]))
    d = rand(i + np.array([1.0, 1.0]))
    
    # Interpolate random numbers
    cubic = f * f * (3.0 - 2.0 * f)

    # Return interpolated value
    value = a + (b - a) * cubic[0] + (c - a) * cubic[1] * (1.0 - cubic[0]) + (d - b) * cubic[0] * cubic[1]

    # Return value
    return value

def noise_arr(xG, yG):
    # Get integer and fractional part of coordinate
    i = np.floor(xG)
    j = np.floor(yG)
    f = np.mod(xG, 1)
    g = np.mod(yG, 1)

    # Get random numbers for 4 corners of cell
    a = rand_arr(i, j)
    b = rand_arr(i + 1, j)
    c = rand_arr(i, j + 1)
    d = rand_arr(i + 1, j + 1)
    
    # Interpolate random numbers
    cubic_x = f * f * (3.0 - 2.0 * f)
    cubic_y = g * g * (3.0 - 2.0 * g)

    # Return interpolated value
    value = a + (b - a) * cubic_x + (c - a) * cubic_y * (1.0 - cubic_x) + (d - b) * cubic_x * cubic_y

    # Return value
    return value

# Function to get fractal Brownian motion
def fractal_brownian_motion(coord):
    # Initialize value and scale
    value = 0.0
    scale = 0.5

    # Loop through octaves
    for i in range(OCTAVES):
        # Add noise to value
        value += noise(coord) * scale

        # Update coordinate and scale for next octave
        coord *= 2.0
        scale *= 0.5
    
    # Return value
    return value

def fractal_brownian_motion_arr(xG, yG):
    # Initialize value and scale
    valueG = np.zeros_like(xG)
    scale = 0.5

    # Loop through octaves
    for i in range(OCTAVES):
        # Add noise to value
        valueG += noise_arr(xG, yG) * scale

        # Update coordinate and scale for next octave
        xG *= 2.0
        yG *= 2.0
        scale *= 0.5
    
    # Return value
    return valueG

# %%
# Coordinates
xs = np.linspace(0, size*4, pixels*2)
ys = np.linspace(0, size*2, pixels)
xG, yG = np.meshgrid(xs, ys)

# %%
# Compare random number generation
t1 = time.time()
randG1 =np.zeros_like(xG)
for i in range(len(ys)):
    for j in range(len(xs)):
        randG1[i, j] = rand(np.array((xs[j], ys[i])))
t2 = time.time()
randG2 = rand_arr(xG, yG)
t3 = time.time()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].imshow(randG1, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
axs[1].imshow(randG2, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
for ax in axs:
    ax.hlines([size], 0, size*4, colors='k', linestyles='dashed')
    ax.vlines([size*2], 0, size*2, colors='k', linestyles='dashed')
axs[0].set_title(f'Time taken: {t2-t1:.2f} s')
axs[1].set_title(f'Time taken: {t3-t2:.2f} s')
plt.show()

# %%
# Compare noise generation
t1 = time.time()
noiseG1 = np.zeros_like(xG)
for i in range(len(ys)):
    for j in range(len(xs)):
        noiseG1[i, j] = noise(np.array((xs[j], ys[i])))
t2 = time.time()
noiseG2 = noise_arr(xG, yG)
t3 = time.time()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].imshow(noiseG1, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
axs[1].imshow(noiseG2, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
for ax in axs:
    ax.hlines([size], 0, size*4, colors='k', linestyles='dashed')
    ax.vlines([size*2], 0, size*2, colors='k', linestyles='dashed')
axs[0].set_title(f'Time taken: {t2-t1:.2f} s')
axs[1].set_title(f'Time taken: {t3-t2:.2f} s')
plt.show()

# %%
# Compare fractal Brownian motion generation
t1 = time.time()
fbmG1 = np.zeros_like(xG)
for i in range(len(ys)):
    for j in range(len(xs)):
        fbmG1[i, j] = fractal_brownian_motion(np.array((xs[j], ys[i])))
t2 = time.time()
fbmG2 = fractal_brownian_motion_arr(xG, yG)
t3 = time.time()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].imshow(fbmG1, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
axs[1].imshow(fbmG2, cmap='Spectral_r', extent=(xs[0], xs[-1], ys[0], ys[-1]))
for ax in axs:
    ax.hlines([size], 0, size*4, colors='k', linestyles='dashed')
    ax.vlines([size*2], 0, size*2, colors='k', linestyles='dashed')
axs[0].set_title(f'Time taken: {t2-t1:.2f} s')
axs[1].set_title(f'Time taken: {t3-t2:.2f} s')
plt.show()

# %%
