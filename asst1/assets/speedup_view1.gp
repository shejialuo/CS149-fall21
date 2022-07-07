set title "the speedup of the thread for view 1"
set xlabel "thread number"
set ylabel "speedup"
plot "speedup_view1" w lp
set terminal pngcairo
set output "The speedup for view 1.png"
replot
set terminal qt
set output