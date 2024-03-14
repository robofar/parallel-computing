plot "<(sed -n '1,1p' ../Assignment03/polygons_large.txt)"using 1:2:(sprintf("Start")) every ::0::0 with labels point pt 7 offset 0,1 notitle, \
"<(sed -n '2,2p' ../Assignment03/polygons_large.txt)"using 1:2:(sprintf("End")) every ::0::0 with labels point pt 7 offset 0,1 notitle, \
"../Assignment03/polygons_large.txt" w filledcurves closed title "polygons", \
'shortest_large.txt' u 1:2:($3-$1):($4-$2) with vectors head filled size 12.0,20,60 lc rgb 'red' title "Shortest path"

