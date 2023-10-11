#!/bin/bash
sim="RL"
objective=("Swimmer-v3")
maxgen=(200)
algorithm=("SNM" "NGA" "EM" "MAP")
gamma=(.8 1 1 .9)
number_runs=5
methods=("evdelta")

rm simulation/reward/*
rm simulation/raw/*

for objec in "${objective[@]}"
do
    for algor in "${!algorithm[@]}"
    do
        g=${gamma[$algor]}
        algo=${algorithm[$algor]}
    python3 parser.py -sim $sim -alg $algo -obj $objec -gen $maxgen -g $g -nruns $number_runs -method $methods
    done
    python3 plot.py -gen $genmax_now -del 0 -sim $sim
    mkdir -p simulation/data/$dim/$objec
    mkdir -p simulation/data/$dim/$objec/quantile
    mv  -v simulation/reward/* simulation/data/$dim/$objec/quantile
    mv  -v simulation/raw/* simulation/data/$dim/$objec
done
exit
