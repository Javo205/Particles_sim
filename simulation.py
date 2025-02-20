import numpy as np


def update(particle, dt):
    particle.position += particle.velocity * dt


def vel_viscossity(particle, viscossity, dt):
    particle.velocity += -viscossity * particle.velocity * dt


def check_collision(particle1, particle2):
    ''' Ellastic collisions '''
    delta_pos = particle2.position - particle1.position
    distance = np.linalg.norm(delta_pos)

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


def check_walls(particle, x_min, x_max, y_min, y_max):
    if particle.position[0] - particle.radius <= x_min and particle.velocity[0] < 0 or particle.position[0] + particle.radius >= x_max and particle.velocity[0] > 0:
        particle.velocity[0] *= -1

    if particle.position[1] - particle.radius <= y_min and particle.velocity[1] < 0 or particle.position[1] + particle.radius >= y_max and particle.velocity[1] > 0:
        particle.velocity[1] *= -1
