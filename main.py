# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from objects import Particle, SpatialGrid
from simulation import check_collision, check_walls, vel_viscossity, Gravitational_forces
from tqdm import tqdm
from utils import load_config

config = load_config()
dt, nframes = config['simulation']['dt'], config['simulation']['nframes']
paredx, paredy = config['physics']['paredx'], config['physics']['paredy']
N, G, viscossity = config['physics']['N'], config['physics']['G'], config['physics']['viscossity']

os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
plot_dir = 'Visuals'
grid = SpatialGrid(cell_size=80)  # paredx / np.sqrt(N))
pbar_sim = tqdm(total=nframes, desc="Simulating Physics")


def update_frame(frame):
    """Update function for animation"""

    pbar_sim.update(1)
    grid.update(particles)

    # Check all pairwise collisions
    for p in particles:
        neighbors = grid.get_nearby_particles(p)

        for other in neighbors:
            if p is not other:
                delta = other.position - p.position
                distance = np.linalg.norm(delta)
                # check_collision(p, other, delta, distance)
                Gravitational_forces(p, other, G, delta, distance, dt)

    for p in particles:
        vel_viscossity(p, viscossity, dt)
        p.update_verlet_scheme(dt)  # Move particles
        # check_walls(p, 0, paredx, 0, paredy)

    for p in particles:
        p.acceleration = np.array([0.0, 0.0])

    # Update circle positions
    for i, p in enumerate(particles):
        circles[i].center = p.position

    return circles  # Return updated circles


# Initialise the particles

center = np.array([paredx / 2, paredy / 2])
radius = np.ones(N)  # + np.array([30, 0, 0])
mass = np.array([3, 3, 3])

pos = np.array([[center[0], center[1]],
                [center[0] - 20, center[0] - 20],
                [center[0] + 30, center[0] - 35]], dtype=float)

vel = np.array([[0, 0],
                [-2, 1],
                [0, 1]], dtype=float)

particles = [Particle(pos[i], vel[i],
                      radius[i], mass[i], dt) for i in range(N)]

# Plotting configuration

box = np.array([[0, 0], [paredx, 0], [paredx, paredy], [0, paredy], [0, 0]])
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, paredx)
ax.set_ylim(0, paredy)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.axis('equal')
# ax.plot(box[:, 0], box[:, 1], 'k', alpha=.6)
circles = [plt.Circle((p.position[0], p.position[1]), p.radius, color='b', fill=True) for p in particles]

for circle in circles:
    ax.add_patch(circle)

ani = animation.FuncAnimation(fig, update_frame, frames=nframes, interval=30, blit=False)
# print('Animation calculated. Saving...')
save_directory = os.path.join(plot_dir)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(os.path.join(plot_dir, 'particles_update.mp4'), writer=writer)
pbar_sim.close()
