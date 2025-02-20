# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from objects import Particle, SpatialGrid
from simulation import update, check_collision, check_walls, vel_viscossity
from config import viscossity, paredx, paredy, dt, N, nframes
from tqdm import tqdm

os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
plot_dir = 'Visuals'
grid = SpatialGrid(cell_size=paredx / np.sqrt(N))
pbar_sim = tqdm(total=nframes, desc="Simulating Physics")


def update_frame(frame):
    """Update function for animation"""

    pbar_sim.update(1)
    grid.update(particles)
    for p in particles:
        vel_viscossity(p, viscossity, dt)
        update(p, dt)  # Move particles
        check_walls(p, 0, paredx, 0, paredy)

    # Check all pairwise collisions
    for p in particles:
        neighbors = grid.get_nearby_particles(p)
        for other in neighbors:
            if p is not other:
                check_collision(p, other)

    # Update circle positions
    for i, p in enumerate(particles):
        circles[i].center = p.position

    return circles  # Return updated circles


# Initialise the particles

radius = np.random.rand(N)
particles = [Particle(np.random.rand() * (paredx - 1 - 2) + 2, np.random.rand() * (paredy - 1 - 2) + 2,
                      np.random.randn() * 3, np.random.randn() * 3, (radius[i] + 0.5) * 1.5, radius[i] * 3) for i in range(N)]

# Plotting configuration

box = np.array([[0, 0], [paredx, 0], [paredx, paredy], [0, paredy], [0, 0]])
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, paredx)
ax.set_ylim(0, paredy)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.axis('equal')
ax.plot(box[:, 0], box[:, 1], 'k', alpha=.6)
circles = [plt.Circle((p.position[0], p.position[1]), p.radius, color='b', fill=True) for p in particles]

for circle in circles:
    ax.add_patch(circle)

ani = animation.FuncAnimation(fig, update_frame, frames=nframes, interval=30, blit=False)
# print('Animation calculated. Saving...')
save_directory = os.path.join(plot_dir)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(os.path.join(plot_dir, 'particles.mp4'), writer=writer)
pbar_sim.close()
