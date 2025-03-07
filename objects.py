import numpy as np


class Particle:
    def __init__(self, pos, vel, radius, mass, dt):
        self.position = pos
        self.velocity = vel
        self.prevpos = self.position - self.velocity * dt
        self.acceleration = np.array([0, 0], dtype=float)
        self.radius = radius
        self.mass = mass

    def update_newton_scheme(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def update_verlet_scheme(self, dt):
        new_position = 2 * self.position - self.prevpos + self.acceleration * dt**2
        self.prev_position = np.copy(self.position)
        self.position = new_position


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
