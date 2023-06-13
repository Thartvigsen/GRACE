#!/bin/bash
################################
# Check the status of the job queue in the cluster
# How to use this script?
# in Cluster Head Node terminal, type: ./slurm_queue.sh
################################

squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.8C %.10m %.11b %R" -u $USER
