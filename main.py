# %%
from simulation import Simulation

if __name__ == "__main__":
    sim = Simulation("config.json")
    sim.save_animation("output.mp4", fps=30)  # Save animation instead of displaying
