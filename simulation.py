import numpy as np
from objects import Particle, SpatialGrid
from utils import load_config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def vel_viscossity(particle, viscossity, dt):
    particle.velocity += -viscossity * particle.velocity * dt


def check_collision(particle1, particle2, delta_pos, distance):
    ''' Ellastic collisions '''

    if distance < particle1.radius + particle2.radius:
        normal = delta_pos / distance
        perp = np.array([-normal[1], normal[0]])

        v1_norm = np.dot(particle1.velocity, normal)
        v1_perp = np.dot(particle1.velocity, perp)
        v2_norm = np.dot(particle2.velocity, normal)
        v2_perp = np.dot(particle2.velocity, perp)

        if v2_norm - v1_norm < 0:
            M = particle1.mass + particle2.mass

            def velsalida(v1, v2, m1, m2):
                return (v1 * (m1 - m2) + 2 * v2 * m2) / M

            u1 = velsalida(v1_norm, v2_norm, particle1.mass, particle2.mass)
            u2 = velsalida(v2_norm, v1_norm, particle2.mass, particle1.mass)

            particle1.velocity = u1 * normal + v1_perp * perp
            particle2.velocity = u2 * normal + v2_perp * perp


def Gravitational_forces(p1, p2, G, delta, distance):

    softening = 1e-2
    if distance > softening:  # Avoid singularity at zero distance
        force_magnitude = G * (p1.mass * p2.mass) / (distance**2 + softening**2)
        force_direction = delta / distance  # Normalize vector
        force = force_magnitude * force_direction

        # Apply Newton's Third Law (equal & opposite forces)
        p1.acceleration += (force / p1.mass)
        p2.acceleration -= (force / p2.mass)  # Opposite direction


class Simulation:
    def __init__(self, config_file="config.json"):
        """Initialize the simulation from a config file."""
        self.config = load_config(config_file)

        # Extract parameters from config
        self.dt = self.config.simulation.dt
        self.nframes = self.config.simulation.nframes
        self.plot_dir = self.config.simulation.plot_dir
        self.paredx = self.config.physics.paredx
        self.paredy = self.config.physics.paredy
        self.N = self.config.physics.N
        self.G = self.config.physics.G
        self.viscosity = self.config.physics.viscosity

        # Initialize particles
        self.particles = self.initialize_particles()

        # Initialize spatial grid for collision handling
        self.grid = SpatialGrid(cell_size=80)

        # 🔹 Setup Matplotlib figure for animation
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.paredx)
        self.ax.set_ylim(0, self.paredy)
        self.circles = [plt.Circle(p.position, p.radius, color='b') for p in self.particles]
        for circle in self.circles:
            self.ax.add_patch(circle)

    def initialize_particles(self):
        """Creates a list of Particle objects with initial conditions."""
        particles = []
        for _ in range(self.N):
            pos = np.random.uniform([0, 0], [self.paredx, self.paredy])
            vel = np.random.uniform([-1, -1], [1, 1])  # Random initial velocity
            mass = np.random.uniform(1, 5)
            radius = 0.1
            particles.append(Particle(pos, vel, radius, mass, self.dt))
        return particles

    def compute_interactions(self):
        """Computes forces on all particles due to gravity."""
        for p in self.particles:
            p.acceleration = np.array([0.0, 0.0])  # Reset acceleration

        for p in self.particles:
            neighbors = self.grid.get_nearby_particles(p)
            for other in neighbors:
                if p is not other:
                    delta = other.position - p.position
                    distance = np.linalg.norm(delta)
                    # Gravitational_forces(p, other, self.G, delta, distance)
                    check_collision(p, other, delta, distance)

    def update_positions(self):
        """Updates particle positions using Verlet integration."""
        for p in self.particles:
            vel_viscossity(p, self.viscosity, self.dt)  # Apply viscosity
            p.update_verlet_scheme(self.dt)  # Move particle
            p.check_walls(0, self.paredx, 0, self.paredy)

    def animate(self, frame):
        """Update function for Matplotlib animation."""
        self.grid.update(self.particles)  # Update spatial grid
        self.compute_interactions()  # Compute forces
        self.update_positions()  # Move particles

        # Update circle positions
        for i, p in enumerate(self.particles):
            self.circles[i].center = p.position

        return self.circles  # Return updated circles

    def save_animation(self, filename="simulation.mp4", fps=30):
        """Saves the simulation as a video instead of displaying it live."""
        print(f"Saving animation to {filename}...")

        # Set up FFMpeg Writer
        Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, metadata=dict(artist="Simulation"), bitrate=1800)

        # Create the animation
        anim = animation.FuncAnimation(self.fig, self.animate, frames=self.nframes, interval=self.dt * 1000, blit=True)

        # Save as a video file
        anim.save(os.path.join(self.plot_dir, filename), writer=writer)

        print(f"Animation saved successfully as {filename}!")
