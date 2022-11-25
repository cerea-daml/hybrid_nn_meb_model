#!/bin/bash

offline_runs="data/wandb/offline-*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync --no-include-synced $ofrun;
    done
    sleep 5m
done
