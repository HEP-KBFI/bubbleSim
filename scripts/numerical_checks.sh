#!/bin/bash

CONF=bubbleSim/config.json

for COMPILER in gcc clang; do
 
    export CXX=$COMPILER
    for OPT in 0 1 2 3; do
        export OPTNAME=O$OPT
        export OPTFLAGS=-O$OPT

        make clean
        make -j4
        
        for DEV in basic pthread; do
            export POCL_DEVICES=$DEV
            build/bubbleSim.exe $CONF > logs/log_${POCL_DEVICES}_${CXX}_${OPTNAME}_1.txt
            build/bubbleSim.exe $CONF > logs/log_${POCL_DEVICES}_${CXX}_${OPTNAME}_2.txt
            build/bubbleSim.exe $CONF > logs/log_${POCL_DEVICES}_${CXX}_${OPTNAME}_3.txt
        done
    done 
done
