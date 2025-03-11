import numpy as np
from objects import Particle, SpatialGrid, KDTreeNeighborSearch
from utils import load_config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm


def compute_optimized_parameters(N, avg_radius, config):
    """
    Dynamically determine optimal search radius and cell size.

    Inputs:
    - N: Number of particles
    - avg_radius: Average radius of particles
    - config: configuration loaded from config.json

    Outputs:
     Depending on the type of simulation and search method, it returns
     the adjusted search parameter, search radius for KDTree search or
     cell_size for Gridsearch
    """
    if N < 500:
        cell_factor = 2.0
    elif N < 5000:
        cell_factor = 3.0
    else:
        cell_factor = 5.0

    search_radius = 2.5 * avg_radius  # For collision detection
    gravity_radius = 50.0 * avg_radius  # For gravitational forces
    cell_size = cell_factor * avg_radius

    if config.physics.search_method == "Grid":
        return cell_size

    elif config.physics.search_method == "KDTree":
        if not config.toggle.gravity_interaction:
            return search_radius
        else:
            return gravity_radius


def avoid_overlap(particle1, particle2, delta_pos, distance):
    """
    Simulation method. Theoretically, it should avoid overlap between
    particle1 and particle2 by separating them along collision axis.

    Inputs:
    - particle1: Particle object
    - paritcle2: Particle object
    - delta_pos: direction of collision
    - distance: module of direction of collision

    Outputs:
     Modification of both positions of particles 1 and 2
    """

    # Check if particles are overlapping
    if distance < particle1.radius + particle2.radius:
        # Calculate normal and tangential directions
        normal = delta_pos / distance
        # Resolve overlap by moving particles apart
        M = particle1.mass + particle2.mass
        overlap = (particle1.radius + particle2.radius - distance)
        move1 = overlap * particle2.mass / M
        move2 = overlap * particle1.mass / M

        # Move particles to avoid overlap
        particle1.position += normal * move1
        particle2.position -= normal * move2


def check_collision_verlet(particle1, particle2, delta_pos, distance):
    """
    Simulation method. Elastic collisions for Verlet integration

    Inputs:
    - particle1: Particle object
    - paritcle2: Particle object
    - delta_pos: direction of collision
    - distance: module of direction of collision

    Outputs:
     Modification of both positions of particles 1 and 2 following an ellastic collision physics
    """
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
    '''
    (not used anymore) Ellastic collisions following euler integration scheme
    '''

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
    """
    Calculation of the acceleration that two interacting particles
    get under gravitational potential
    """
    softening = max(p1.radius + p2.radius, 0.1)
    if distance > softening:  # Avoid singularity at zero distance
        force_magnitude = G * (p1.mass * p2.mass) / (distance**2 + softening**2)
        force_direction = delta / distance  # Normalize vector
        force = force_magnitude * force_direction

        # Apply Newton's Third Law (equal & opposite forces)
        p1.acceleration += (force / p1.mass)
        p2.acceleration -= (force / p2.mass)  # Opposite direction


class Simulation:
    """
    Object that initializes all needed to compute the simulation and
    covers the functions that generate and save the animation.
    """
    def __init__(self, config_file="config.json"):
        """Initialize the simulation from a config file."""
        self.config = load_config(config_file)

        # Extract parameters from config
        self.dt = self.config.simulation.dt
        self.total_animation_frames = self.config.simulation.nframes // self.config.simulation.physics_steps_per_frame

        self.plot_dir = self.config.simulation.plot_dir
        self.paredx = self.config.physics.paredx
        self.paredy = self.config.physics.paredy
        self.N = self.config.physics.N
        self.G = self.config.physics.G
        self.g = self.config.physics.g
        self.viscosity = self.config.physics.viscosity
        self.search_method = self.config.physics.search_method
        self.gravity_interaction = self.config.toggle.gravity_interaction
        self.wall_interaction = self.config.toggle.wall_interaction
        self.particle_initialization = self.config.particle_initialization
        self.max_velocity_seen = 0.01

        # Initialize particles
        self.particles = self.initialize_particles()
        self.search_parameter = compute_optimized_parameters(self.N, self.rad_sum / self.N, self.config)

        # Initialize spatial grid for collision handling
        if self.search_method == "Grid":
            self.grid = SpatialGrid(cell_size=self.search_parameter)

        elif self.search_method == "KDTree":
            self.grid = KDTreeNeighborSearch(self.particles, self.search_parameter)

        # Setup Matplotlib figure for animation
        self.pbar_sim = tqdm(total=self.total_animation_frames, desc="Simulating Physics")
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
        self.circles = [plt.Circle(p.position, p.radius, color=p.color) for p in self.particles]
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
            self.g = 0
            self.search_method = 'KDTree'
            radius = np.array([10, 1, 0.1])
            mass = np.array([70, 1, 0.1])
            pos = np.array([[self.paredx / 2, self.paredy / 2],
                            [self.paredx / 2 - 22, self.paredy / 2 - 22],
                            [self.paredx / 2 - 24.2, self.paredy / 2 - 24.2]])
            vel = np.array([[0, 0],
                            [-4, 4],
                            [-5.7, 5.73]])
            for i in range(3):
                particles.append(Particle(pos[i], vel[i], radius[i], mass[i], self.dt))

            self.rad_sum = np.sum(radius)

        elif self.particle_initialization == "gravitation2":
            self.gravity_interaction = 1
            self.wall_interaction = 0
            self.N = 3
            self.paredx = 500
            self.paredy = 500
            self.g = 0
            radius = np.array([10, 0.5, 10])
            mass = np.array([50, 0.1, 50])
            pos = np.array([[self.paredx / 2 + 20, self.paredy / 2],
                            [self.paredx / 2 - 35, self.paredy / 2 - 35],
                            [self.paredx / 2 - 20, self.paredy / 2]])
            vel = np.array([[0, 2.5],
                            [-4, 4],
                            [0, -2.6]])
            for i in range(3):
                particles.append(Particle(pos[i], vel[i], radius[i], mass[i], self.dt))

            self.rad_sum = np.sum(radius)

        elif self.particle_initialization == "grid":
            rows = int(np.sqrt(self.N))  # Approximate number of rows
            cols = int(np.ceil(self.N / rows))  # Compute columns to fit N

            # Compute spacing between particles
            x_spacing = self.paredx / (cols + 1)  # Leave margins
            y_spacing = self.paredy / (rows + 1)

            # Assign positions
            index = 0
            for i in range(rows):
                for j in range(cols):
                    if index >= self.N:
                        break  # Stop if we placed all N particles

                    # Compute position (avoid boundaries)
                    x_pos = (j + 1) * x_spacing
                    y_pos = (i + 1) * y_spacing

                    radius = np.random.uniform(0.5, 1.5) * 2  # Set radius
                    mass = radius * np.random.uniform(2, 2.5)  # Set mass
                    vel = np.random.uniform([-1, -1], [1, 1]) * 10  # Initial velocity

                    particles.append(Particle(np.array([x_pos, y_pos]), vel, radius, mass, self.dt))
                    index += 1

            self.rad_sum = sum(p.radius for p in particles)  # Update radius sum
        return particles

    def compute_interactions(self, p):
        """Computes interactions particle - particle."""
        neighbors = self.grid.get_nearby_particles(p)
        for other in neighbors:
            if p is not other:
                delta = other.position - p.position
                distance = np.linalg.norm(delta)
                if self.gravity_interaction:
                    Gravitational_forces(p, other, self.G, delta, distance)

                check_collision_verlet(p, other, delta, distance)
                # only_overlap(p, other, delta, distance)

    def update_positions(self):
        """Updates particle positions using Verlet integration."""
        for p in self.particles:
            p.acceleration = np.array([0.0, 0.0])  # Reset acceleration

        for p in self.particles:
            p.vel_viscosity(self.viscosity, self.dt)  # Apply viscosity
            p.add_Gravity(self.g)  # Sum gravity forces
            if self.wall_interaction:
                p.check_walls(0, self.paredx, 0, self.paredy)  # Wall constraints
            self.compute_interactions(p)  # Compute p - p interaction
            p.update_verlet_scheme(self.dt)  # Move particle

    def animate(self, frame):
        """Update function for Matplotlib animation. (not used anymore)"""
        for _ in range(self.config.simulation.physics_steps_per_frame):
            self.grid.update(self.particles)
            self.update_positions()

        # Update circle positions
        velocities = self.velocities_norm()

        for i, p in enumerate(self.particles):
            p.update_color(velocities[i])
            self.circles[i].center = p.position
            self.circles[i].set_color(p.color)

        self.pbar_sim.update(1)

        return self.circles  # Return updated circles

    def save_animation(self, filename="simulation.mp4", fps=30):
        """Saves the simulation as a video instead of displaying it live."""
        print(f"Saving animation to {filename}...")

        # Set up FFMpeg Writer
        Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, metadata=dict(artist="Simulation"), bitrate=1800)

        # Perform a number of substep simuation frames before rendering
        def animate_wrapper(frame):
            for _ in range(self.config.simulation.physics_steps_per_frame):
                self.grid.update(self.particles)
                self.update_positions()

            # Update circle positions
            for i, p in enumerate(self.particles):
                vel = p.estimated_velocity_module()
                self.max_velocity_seen = max(self.max_velocity_seen, vel)
                normalized_vel = vel / self.max_velocity_seen
                p.update_color(normalized_vel)
                self.circles[i].center = p.position
                self.circles[i].set_color(p.color)

            self.pbar_sim.update(1)
            return self.circles

        anim = animation.FuncAnimation(
            self.fig,
            animate_wrapper,
            frames=self.total_animation_frames,
            interval=self.dt * self.config.simulation.physics_steps_per_frame * 1000,
            blit=True
        )

        # Save as video file
        anim.save(os.path.join(self.plot_dir, filename), writer=writer)
        self.pbar_sim.close()
        print(f"Animation saved successfully as {filename}!")
