from simulation import Simulation

# TODO: Document project and write README
if __name__ == "__main__":
    sim = Simulation("config.json")  # Initialization of Simulation object
    sim.save_animation("output.mp4", fps=30)  # Save animation
