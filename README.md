# Particle Simulation

This project simulates colliding particles using Verlet integration, elastic collisions, and optional gravity interactions.
It was created as a personal project to deepen my understanding of Object-Oriented Programming (OOP) and version control.
It is continuously evolving, with planned improvements and potential extensions into related physics-based simulations, which may emerge as independent projects in the future.
AI has been a valuable tool in its development—currently, I focus more on understanding concepts rather than writing code from scratch.


## Features
- Supports **SpatialGrid and KDTree** for neighbor searches  
- **Configurable settings** via `config.json`  
- **Progress bar** with `tqdm`  
- **Verlet integration** for stable motion  
- Saves output as `.mp4` instead of GUI display  

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the simulation:
   ```bash
   python main.py
3. Adjust settings in `config.json` to modify the simulation parameters.

## Configuration options
The `config.json` file allows customization of the simulation
   ```json
{
    "simulation":{
        "dt": 0.005,
        "nframes": 5000,
        "physics_steps_per_frame": 10,
        "plot_dir": "Visuals"
    },
    "physics":{
        "paredx": 100,
        "paredy": 100,
        "N": 40,
        "G": 10,
        "g": -7,
        "viscosity": 0,
        "search_method": "Grid" 
    },
    "toggle":{
        "gravity_interaction": 0,
        "wall_interaction": 1
    },
    "particle_initialization":"grid"
    
}
```
### Key parameters
- `search_method`: Choose `Grid` or `KDTree` for neighbor search optimization.
- `gravity_interaction`: Set to `1` to enable gravity, `0` to disable.
- `wall_interaction`: Set to `1` to enable wall collisions, `0` to disable.
- `N`: number of particles in the simulation
- `particle_initialization`: can take `random`, `grid`, `gravity1` or `gravity2` for different case scenarios. It is suggested that for the last two, `gravity_interaction` must be toggled in order for it to work properly

## Future Improvements
- Improve Code Readability & Documentation
    - Add inline comments and refactor code for clarity. (DONE ALMOST)
- Enhance Particle Initialization
    - Support different initialization scenarios depending on interactions (e.g., clustered groups, uniform distribution, random velocities). (DONE, maybe it is better to modularize the scenarios)
- Add Lennard-Jones Potential
    - Implement Lennard-Jones interactions to simulate solid-like behavior in molecular dynamics.
- Enable Mixed Interactions
    - Allow different particle types to have customized interaction rules within the same simulation.
