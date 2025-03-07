import numpy as np
from objects import Particle, SpatialGrid
from utils import load_config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def vel_viscossity(particle, viscossity, dt):
    particle.velocity += -viscossity * particle.velocity * dt


def check_collision_verlet(particle1, particle2, delta_pos, distance):
    """Elastic collisions for Verlet integration"""
    # Check if particles are overlapping
    if distance < particle1.radius + particle2.radius:
        # Calculate normal and tangential directions
        normal = delta_pos / distance
        tangent = np.array([-normal[1], normal[0]])

        # Calculate current velocities using position differences
        v1 = particle1.estimated_velocity()
        v2 = particle2.estimated_velocity()

        # Project velocities onto normal and tangential directions
        v1_normal = np.dot(v1, normal)
        v1_tangent = np.dot(v1, tangent)
        v2_normal = np.dot(v2, normal)
        v2_tangent = np.dot(v2, tangent)

        # Only process collision if particles are approaching each other
        if v2_normal - v1_normal < 0:
            # Calculate total mass
            M = particle1.mass + particle2.mass

            # Calculate new normal velocities using conservation of momentum and energy
            new_v1_normal = ((particle1.mass - particle2.mass) * v1_normal + 2 * particle2.mass * v2_normal) / M
            new_v2_normal = ((particle2.mass - particle1.mass) * v2_normal + 2 * particle1.mass * v1_normal) / M

            # Resolve overlap by moving particles apart
            overlap = (particle1.radius + particle2.radius - distance)
            move1 = overlap * particle2.mass / M
            move2 = overlap * particle1.mass / M

            # Move particles to avoid overlap
            particle1.position = particle1.position + normal * move1
            particle2.position = particle2.position - normal * move2

            # Update previous positions to reflect new velocities
            # For normal component
            particle1.prev_position = particle1.position - (
                new_v1_normal * normal + v1_tangent * tangent) * particle1.dt
            particle2.prev_position = particle2.position - (
                new_v2_normal * normal + v2_tangent * tangent) * particle2.dt


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
        box = np.array([[0, 0], [self.paredx, 0], [self.paredx, self.paredy], [0, self.paredy], [0, 0]])
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.paredx)
        self.ax.set_ylim(0, self.paredy)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')
        self.ax.axis('equal')
        self.ax.plot(box[:, 0], box[:, 1], 'k', alpha=.6)
        self.circles = [plt.Circle(p.position, p.radius, color='b') for p in self.particles]
        for circle in self.circles:
            self.ax.add_patch(circle)

    def initialize_particles(self):
        """Creates a list of Particle objects with initial conditions."""
        particles = []
        for _ in range(self.N):
            radius = np.random.uniform(0.5, 2.5)
            mass = radius * 4
            pos = np.random.uniform([radius, radius], [self.paredx - radius, self.paredy - radius])
            vel = np.random.uniform([-1, -1], [1, 1])  # Random initial velocity
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
                    check_collision_verlet(p, other, delta, distance)

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
