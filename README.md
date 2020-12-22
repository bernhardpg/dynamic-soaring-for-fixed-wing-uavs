# Dynamic Soaring for Fixed-Wing UAVs

My specialization project during my second last semester at NTNU.

This code generates dynamic soaring trajectories for fixed-wing UAVs, in any directions towards the wind. The problem is formulated as an optimal control problems, and trajectories are generated using direct collocation. 

An example of trajectories generated at all angles towards the wind:

![](trajectories.gif)

## To generate trajectories
Run the following to generate a trajectory for angle `a`

```./main.py -a <angle>```

To run a sweep search for all angles, run

```./main.py -s <n_sweep_angles>```
where ```n_sweep_angles``` specifies the number of angles that will be generated.

The full set of options is:

```./main.py -a <angle> -p <period_guess> -v <velocity_guess> -s <n_sweep_angles>```

For more details, see the file 'dynamic_soaring.pdf'.
