plot "<(sed -n '1,1p' polygons_large.txt)"using 1:2:(sprintf("Start")) every ::0::0 with labels point pt 7 offset 0,1 notitle, \
"<(sed -n '2,2p' polygons_large.txt)"using 1:2:(sprintf("End")) every ::0::0 with labels point pt 7 offset 0,1 notitle, \
"polygons_large.txt" w filledcurves closed title "polygons", \
'vg_large.txt' u 1:2:($3-$1):($4-$2) with vectors head filled size 0.2,20,60 title "vis graph"