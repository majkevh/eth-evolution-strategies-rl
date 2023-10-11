#!/bin/bash
sim="BM"
objective=("Hypersphere")
dimension=(2)
maxgen=(1500)
population=(20)
algorithm=("SNM")
gamma=(1)
number_runs=5
methods=("evdelta") 
type_cov=("full")
rm simulation/error/*
rm simulation/raw/*

for objec in "${objective[@]}"
do
    for dims in "${!dimension[@]}"
    do
        dim=${dimension[$dims]}
        pop=${population[$dims]}
        genmax_now=${maxgen[$dims]}
        for algor in "${!algorithm[@]}"
        do
            g=${gamma[$algor]}
            algo=${algorithm[$algor]}
        python3 parser.py -sim $sim -alg $algo -obj $objec -dim $dim -gen $genmax_now -pop $pop -g $g -nruns $number_runs -method $methods -cov $type_cov
        done
        python3 plot.py -gen $genmax_now -del 0 -sim $sim
        mkdir -p simulation/data/$type_cov/$dim/$objec
        mkdir -p simulation/data/$type_cov/$dim/$objec/quantile
        mv  -v simulation/error/* simulation/data/$type_cov/$dim/$objec/quantile
        mv  -v simulation/raw/* simulation/data/$type_cov/$dim/$objec
    done
done
exit
