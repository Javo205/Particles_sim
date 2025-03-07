import numpy as np


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

    def check_walls(self, x_min, x_max, y_min, y_max):
        vx = self.estimated_velocity()[0]
        vy = self.estimated_velocity()[1]

        # Left wall collision
        if self.position[0] <= x_min + self.radius and self.prev_position[0] > x_min:
            self.position[0] = x_min + self.radius
            self.prev_position[0] = self.position[0] + (self.position[0] - self.prev_position[0])  # Reverse direction

        # Right wall collision
        if self.position[0] + self.radius >= x_max and self.prev_position[0] < x_max:
            self.position[0] = x_max - self.radius
            self.prev_position[0] = self.position[0] + (self.position[0] - self.prev_position[0])  # Reverse direction

        # Bottom wall collision
        if self.position[1] - self.radius <= y_min and self.prev_position[1] > y_min:
            self.position[1] = y_min + self.radius
            self.prev_position[1] = self.position[1] + (self.position[1] - self.prev_position[1])  # Reverse direction

        # Top wall collision
        if self.position[1] + self.radius >= y_max and self.prev_position[1] < y_max:
            self.position[1] = y_max - self.radius
            self.prev_position[1] = self.position[1] + (self.position[1] - self.prev_position[1])  # Reverse direction


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
