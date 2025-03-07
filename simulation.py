import numpy as np


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


def Gravitational_forces(p1, p2, G, delta, distance, dt):

    softening = 1e-2
    if distance > softening:  # Avoid singularity at zero distance
        force_magnitude = G * (p1.mass * p2.mass) / (distance**2 + softening**2)
        force_direction = delta / distance  # Normalize vector
        force = force_magnitude * force_direction

        # Apply Newton's Third Law (equal & opposite forces)
        p1.acceleration += (force / p1.mass)
        p2.acceleration -= (force / p2.mass)  # Opposite direction
