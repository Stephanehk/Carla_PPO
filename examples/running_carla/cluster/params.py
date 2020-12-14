import pathlib


# change these.
CLUSTER_QUICKSTART_DIR = '/u/bzhou/cluster_quickstart'
CARLA_DIR = '/scratch/cluster/bzhou/software/CARLA_0.9.10.1'

HEADER = f"""#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

export WORLD_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)
export TM_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)
export PYTHONPATH={CARLA_DIR}/PythonAPI/carla/dist/*py3*

sh {CARLA_DIR}/CarlaUE4.sh -world-port=$PORT -opengl -quality-level=Epic &

sleep 30
"""

BODY = """
cd {target_dir}
python3 run.py --world_port $WORLD_PORT --tm_port $WORLD_PORT --n_vehicles {n_vehicles}
"""

FOOTER = """
ps
kill 0
"""


PARAMS = {
        'n_cars': [10, 50],
        }


def get_job(n_cars):
    body = BODY.format(
            target_dir=pathlib.Path(__file__).parent.parent,
            n_vehicles=n_vehicles)

    return HEADER + body + FOOTER
