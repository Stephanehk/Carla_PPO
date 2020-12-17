import pathlib


# change these.
CLUSTER_QUICKSTART_DIR = '/scratch/cluster/stephane/cluster_quickstart'
CARLA_DIR = '/scratch/cluster/stephane/Carla_0.9.10'

HEADER = f"""#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

export WORLD_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)
export TM_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)

export CARLA_EGG=$(ls {CARLA_DIR}/PythonAPI/carla/dist/*py3*)
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG

echo $WORLD_PORT
echo $TM_PORT

sh {CARLA_DIR}/CarlaUE4.sh -world-port=$WORLD_PORT -opengl -quality-level=Epic &

sleep 30
"""

BODY = """
cd {target_dir}
python3 run_ppo.py --world_port $WORLD_PORT --tm_port $TM_PORT --n_vehicles {n_vehicles}
"""

FOOTER = """
ps
kill 0
"""


PARAMS = {
        'n_vehicles': [5, 10],
        }


def get_job(n_vehicles):
    body = BODY.format(
            target_dir=pathlib.Path(__file__).parent.parent,
            n_vehicles=n_vehicles)

    return HEADER + body + FOOTER
