### SSH

`ssh -X -v stephane@hypnotoad.cs.utexas.edu`

### Storage

All files (inlcuding this directory) are kept in `/scratch/cluster/stephane`. 

## Running CARLA 

The path to Carla is set in `examples/running_carla/cluster/params.py`. This script to run (ie: carla_ppo.py) can be set here as well. 

```bash
python3 generate_slurm.py examples/running_carla/cluster/params.py

cd examples/running_carla/cluster/slurm_scripts

sbatch n_vehicles=10.submit
```

then you can look at the running jobs using `squeue -u stephane`.  

When your job is running/done you'll be able to check out the `examples/running_carla/cluster/logs` and see the individual STDERR, STDOUT.

## Scripts
`examples/running_carla/run_ppo` - PPO for collision avoidance
`examples/running_carla/run_ppo_navigation` - PPO to navigate from a start point to an end point
`examples/running_carla/statistics_manager` - Helper script to calculate CARLA score

