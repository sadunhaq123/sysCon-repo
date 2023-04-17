#!/bin/bash

container_name=$1
time_in_seconds=$2
logfile_name=$3

sudo sysdig container.id=$(docker container ls | grep "$container_name" | awk '{print $1}') -M "$time_in_seconds" -w "$logfile_name"



#bash sysdig_commands.sh my_ubuntu 30 my_ubuntu.log 
#This implies that sysdif will collect traces for the container named 'my_ubuntu' for 30s and output to a my_ubuntu.log file.
