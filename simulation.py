import numpy as np
from objects import Particle, SpatialGrid, KDTreeNeighborSearch
from utils import load_config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm


def compute_optimized_parameters(N, avg_radius, config):
    """Dynamically determine optimal search radius and cell size."""
    if N < 500:
        cell_factor = 2.0
    elif N < 5000:
        cell_factor = 3.0
    else:
        cell_factor = 5.0

    search_radius = 2.5 * avg_radius  # For collision detection
    gravity_radius = 10.0 * avg_radius  # For gravitational forces
    cell_size = cell_factor * avg_radius

    if config.physics.search_method == "Grid":
        return cell_size

    elif config.physics.search_method == "KDTree":
        if not config.toggle.gravity_interaction:
            return search_radius
        else:
            return gravity_radius


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


# TODO: Add LJ potential and see what happens
def Gravitational_forces(p1, p2, G, delta, distance):

    softening = max(p1.radius + p2.radius, 0.1)
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
        self.search_method = self.config.physics.search_method
        self.gravity_interaction = self.config.toggle.gravity_interaction
        self.wall_interaction = self.config.toggle.wall_interaction
        self.particle_initialization = self.config.particle_initialization

        # Initialize particles
        self.particles = self.initialize_particles()
        self.search_parameter = compute_optimized_parameters(self.N, self.rad_sum / self.N, self.config)

        # Initialize spatial grid for collision handling
        if self.search_method == "Grid":
            self.grid = SpatialGrid(cell_size=self.search_parameter)

        elif self.search_method == "KDTree":
            self.grid = KDTreeNeighborSearch(self.particles, self.search_parameter)

        # Setup Matplotlib figure for animation
        self.pbar_sim = tqdm(total=self.nframes, desc="Simulating Physics")
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.paredx)
        self.ax.set_ylim(0, self.paredy)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')
        self.ax.axis('equal')
        if self.wall_interaction:
            box = np.array([[0, 0], [self.paredx, 0], [self.paredx, self.paredy], [0, self.paredy], [0, 0]])
            self.ax.plot(box[:, 0], box[:, 1], 'k', alpha=.6)
        self.circles = [plt.Circle(p.position, p.radius, color='b') for p in self.particles]
        for circle in self.circles:
            self.ax.add_patch(circle)

    # TODO: Particle initialization function creating different scenarios, not just random
    def initialize_particles(self):
        """Creates a list of Particle objects with initial conditions."""
        particles = []

        if self.particle_initialization == "random":
            self.rad_sum = 0
            for i in range(self.N):
                radius = np.random.uniform(0.5, 1.5) * 30
                mass = radius * np.random.uniform(2, 2.5)
                pos = np.random.uniform([radius, radius], [self.paredx - radius, self.paredy - radius])
                vel = np.random.uniform([-1, -1], [1, 1]) * 30  # Random initial velocity
                particles.append(Particle(pos, vel, radius, mass, self.dt))
                self.rad_sum += radius

        elif self.particle_initialization == "gravitation1":
            self.gravity_interaction = 1
            self.wall_interaction = 0
            self.N = 3
            self.paredx = 100
            self.paredy = 100
            radius = np.array([10, 1, 0.1])
            mass = np.array([70, 1, 0.06])
            pos = np.array([[self.paredx / 2, self.paredy / 2],
                            [self.paredx / 2 - 22, self.paredy / 2 - 22],
                            [self.paredx / 2 - 24.2, self.paredy / 2 - 24.2]])
            vel = np.array([[0, 0],
                            [-4, 4],
                            [-5.6, 5.6]])
            for i in range(3):
                particles.append(Particle(pos[i], vel[i], radius[i], mass[i], self.dt))

            self.rad_sum = np.sum(radius)

        elif self.particle_initialization == "gravitation2":
            self.gravity_interaction = 1
            self.wall_interaction = 0
            self.N = 3
            self.paredx = 100
            self.paredy = 100
            radius = np.array([10, 1, 1])
            mass = np.array([70, 1, 1])
            pos = np.array([[self.paredx / 2, self.paredy / 2],
                            [self.paredx / 2 - 15, self.paredy / 2 - 15],
                            [self.paredx / 2, self.paredy / 2 - 25]])
            vel = np.array([[0, 0],
                            [-5, 5],
                            [5, 0]])
            for i in range(3):
                particles.append(Particle(pos[i], vel[i], radius[i], mass[i], self.dt))

            self.rad_sum = np.sum(radius)
        return particles

    def compute_interactions(self):
        """Computes interactions particle - particle."""
        for p in self.particles:
            p.acceleration = np.array([0.0, 0.0])  # Reset acceleration

        for p in self.particles:
            neighbors = self.grid.get_nearby_particles(p)
            for other in neighbors:
                if p is not other:
                    delta = other.position - p.position
                    distance = np.linalg.norm(delta)
                    if self.gravity_interaction:
                        Gravitational_forces(p, other, self.G, delta, distance)

                    check_collision_verlet(p, other, delta, distance)

    def update_positions(self):
        """Updates particle positions using Verlet integration."""
        for p in self.particles:
            p.vel_viscossity(self.viscosity, self.dt)  # Apply viscosity
            p.update_verlet_scheme(self.dt)  # Move particle
            if self.wall_interaction:
                p.check_walls(0, self.paredx, 0, self.paredy)

    def animate(self, frame):
        """Update function for Matplotlib animation."""
        self.grid.update(self.particles)  # Update spatial grid
        self.compute_interactions()  # Compute forces
        self.update_positions()  # Move particles

        # Update circle positions
        for i, p in enumerate(self.particles):
            self.circles[i].center = p.position

        self.pbar_sim.update(1)

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
        self.pbar_sim.close()
        print(f"Animation saved successfully as {filename}!")
