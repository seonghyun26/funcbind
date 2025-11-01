#!~/micromamba/envs/funcbind/bin/python

import argparse
from datetime import datetime
import os
import time

##############################
# parse args
parser = argparse.ArgumentParser("funcbind", add_help=False)
parser.add_argument("--job_template", default="job_template_1gpu.sbatch", type=str)
parser.add_argument("--wandb_agent", default=None, type=str)
parser.add_argument("--n_jobs", default=1, type=int)
args = parser.parse_args()

##############################
# read sbatch job template
with open(args.job_template) as fp:
    sbatch_lines = fp.readlines()

##############################
# add python script line
sbatch_lines.append(f"srun wandb agent --count 1 {args.wandb_agent} ")

##############################
# write sbatch
dt = datetime.now().strftime('%Y%m%d_%H%M%S_agent').replace("'", '')
os.makedirs("~/tmp/logs", exist_ok=True)
fname = f"~/tmp/logs/{dt}.sbatch"
with open(fname, "w") as fp:
    for line in sbatch_lines:
        fp.write(line)

print(f"Generated sbatch file: {fname}")

##############################
# execute sbatch
for i in range(args.n_jobs):
    print(f"Submitting job {i+1}/{args.n_jobs}")
    os.system(f"sbatch {fname}")
    time.sleep(1)