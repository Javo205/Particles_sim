import numpy as np
import scipy.spatial as spatial


# TODO: change particle color based on speed
class Particle:
    def __init__(self, pos, vel, radius, mass, dt):
        self.position = pos
        self.velocity = vel
        self.prev_position = self.position - self.velocity * dt
        self.acceleration = np.array([0, 0], dtype=float)
        self.radius = radius
        self.mass = mass
        self.dt = dt

    def estimated_velocity(self):
        """Estimate velocity using finite differences (Verlet-compatible)."""
        return (self.position - self.prev_position) / self.dt

    def update_newton_scheme(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def update_verlet_scheme(self, dt):
        new_position = 2 * self.position - self.prev_position + self.acceleration * self.dt**2
        self.prev_position = np.copy(self.position)
        self.position = new_position

    def vel_viscossity(self, viscossity, dt):
        velocity = (self.position - self.prev_position) / self.dt
        velocity = -viscossity * velocity
        self.prev_position = self.prev_position - velocity * self.dt

    def check_walls(self, x_min, x_max, y_min, y_max):
        """Handle wall collisions with Verlet integration"""
        # Calculate current velocity from positions
        damping = 1.0
        velocity = (self.position - self.prev_position) / self.dt

        # Left wall collision
        if self.position[0] - self.radius <= x_min:
            # Ensure we were coming from inside
            if self.prev_position[0] - self.radius > x_min:
                # Place exactly at boundary
                self.position[0] = x_min + self.radius
                # Reverse x component of velocity
                velocity[0] = -velocity[0] * damping  # Adding a damping factor of damping
                # Recalculate previous position based on new velocity
                self.prev_position[0] = self.position[0] - velocity[0] * self.dt

        # Right wall collision
        if self.position[0] + self.radius >= x_max:
            if self.prev_position[0] + self.radius < x_max:
                self.position[0] = x_max - self.radius
                velocity[0] = -velocity[0] * damping
                self.prev_position[0] = self.position[0] - velocity[0] * self.dt

        # Bottom wall collision
        if self.position[1] - self.radius <= y_min:
            if self.prev_position[1] - self.radius > y_min:
                self.position[1] = y_min + self.radius
                velocity[1] = -velocity[1] * damping
                self.prev_position[1] = self.position[1] - velocity[1] * self.dt

        # Top wall collision
        if self.position[1] + self.radius >= y_max:
            if self.prev_position[1] + self.radius < y_max:
                self.position[1] = y_max - self.radius
                velocity[1] = -velocity[1] * damping
                self.prev_position[1] = self.position[1] - velocity[1] * self.dt


class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell(self, position):
        # Compute cell indices based on particle position.
        # Adjust this if your simulation includes negative coordinates.
        return (int(position[0] // self.cell_size), int(position[1] // self.cell_size))

    def add_particle(self, particle):
        # Add particle to the appropriate cell.
        cell = self._get_cell(particle.position)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(particle)

    def get_nearby_particles(self, particle):
        # Retrieve all particles in the cell of the particle and the 8 surrounding cells.
        x, y = self._get_cell(particle.position)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (x + dx, y + dy)
                if cell in self.grid:
                    nearby.extend(self.grid[cell])
        return nearby

    def update(self, particles):
        # Clear and rebuild the grid. This should be done every simulation step.
        self.grid.clear()
        for p in particles:
            self.add_particle(p)


# TODO: ckdtree maybe gives better performance CHECK
class KDTreeNeighborSearch:
    def __init__(self, particles, search_radius):
        """ Initializes the KDTree using particle positions"""
        self.search_radius = search_radius
        self.update(particles)

    def update(self, particles):
        """ Rebuilds the KDTree """
        self.particles = particles
        positions = np.array([p.position for p in particles])
        self.tree = spatial.cKDTree(positions)

    def get_nearby_particles(self, particle):
        """ Returns nearby particles using KDTree's fast query"""
        indices = self.tree.query_ball_point(particle.position, self.search_radius)
        return [self.particles[i] for i in indices if self.particles[i] is not particle]
