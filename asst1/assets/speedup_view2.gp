set title "the speedup of the thread for view 2"
set xlabel "thread number"
set ylabel "speedup"
plot "speedup_view2" w lp
set terminal pngcairo
set output "The speedup for view 2.png"
replot
set terminal qt
set output