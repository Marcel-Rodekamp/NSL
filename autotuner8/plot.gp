#!/bin/env gnuplot
set term svg enhanced background rgb 'white' size 1295,800 #1920,1080;
set output "autotuning_graph.svg"
set border lw 0.5
set multiplot layout 2,1

set xlabel font "Liberation Serif, " 
set ylabel font "Liberation Serif, " 
set xtics font "Liberation Serif, " 
set ytics font "Liberation Serif, " 
set title font "Liberation Serif, "
set key font "Liberation Serif, "
set key bottom right
set xlabel "Number of trajectories"
set ylabel "Acceptance Rate"

set yrange [0:1]

in = "<cat autotuner.out|grep traj| tr '/' ' '| tr ')' ' '| tr '(' ' '| tr ',' ' '"

plot in u 2:5 w lp lt 2 ps 0.6 t "Overall Acceptance Rate (binomial)",in u 2:7 w lp lt 3 ps 0.6 t "Overall Acceptance Rate (probabilities)", in."|awk '{tr=$13; if(tr==1){print lastline}; {lastline=$0}}'" u 2:17 lt 2 pt 4 t "Acceptance Rate for current steps (binomial)", in."|awk '{tr=$13; if(tr==1){print lastline}; {lastline=$0}}'" u 2:22:24:25 w errorbars lt 3 pt 5 t "Acceptance Rate for current steps (probabilities)", 0.66 dashtype 2 lc rgb "0xbbbbbb"



unset yrange
set ylabel "N_{md}"
set logscale y

plot in u 2:11 lt 6 ps 0.4 t ""

unset multiplot
q
