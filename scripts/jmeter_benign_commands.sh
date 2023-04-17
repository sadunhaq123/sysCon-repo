#!/bin/bash


directory=$1
load_file=$2
log_file=$3

cd $directory
./jmeter -n -t $load_file -l $log_file


#cd /home/sadun/Downloads/apache-jmeter-5.5/bin
#./jmeter -n -t seven_minutes.jmx -l seven_1.log

#This file has 3 command line parameters. First argument is to traverse to the folder where jmeter is installed. 
#Second argument is the name of the load file. Third argument is the file for logging purposes.
# The example shows traversing to the jmeter folder, executing the load file named 'seven_minutes.jmx' and logging to the file 'seven_1.log'
