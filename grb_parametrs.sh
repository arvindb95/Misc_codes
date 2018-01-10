#!/bin/sh

# For visual_test
python visual_test.py "GRB160131A" 191924430 110 1.23 -2.5 1125 2.9e-3 --pixbin 8 --dt_background_left 200 --dt_background_right 200
python visual_test.py "GRB160325A" 196585150 60 -0.64 -2.42 232.7 0.03 --pixbin=8 --sim_scale 20
python visual_test.py "GRB160703A" 205243800 40 1.21 -2.5 327 2.62e-3 --pixbin=8 --dt_background_left 200 --dt_background_right 200
python visual_test.py "GRB160910A" 211223975 30 -0.36 -2.38 329.99 0.23 --pixbin=8 --dt_background_left 200 --dt_background_right 200
python visual_test.py "GRB160509A" 200480320 45 -0.75 -2.13 333.59 0.15 --pixbin 8 --dt_background_left 80 --dt_background_right 100
python visual_test.py "GRB160607A" 202994020 380 0.89 -2.5 131.48 5.07e-3 --pixbin 8 --dt_background_left 10 --dt_background_right 200
python visual_test.py "GRB160623A" 204353975 90 -0.5 -2.5 562 5e-3 --pixbin 8 --dt_background_left 200 --dt_background_right 100
python visual_test.py "GRB160821A" 209507780 43 -0.97 -2.25 865.96 0.15 --pixbin 8 --dt_background_left 200 --dt_background_right 200
python visual_test.py "GRB151006A" 181821280 90 -1.1 -1.8 218 5e-3 --pixbin 16 --dt_background_left 200 --dt_background_right 400
# For 3d_plotting
python 3d_transient_loc.py "GRB160131A" 191924435 78.168 -7.05 116.86 184.86
python 3d_transient_loc.py "GRB160325A" 196585150 16.0025 -72.048 0.66 159.48
python 3d_transient_loc.py "GRB160509A" 200480320 311.3 76.1 105.66 85.52
python 3d_transient_loc.py "GRB160607A" 202994020 13.667 -4.95 138.85 315.78
python 3d_transient_loc.py "GRB160623A" 204353975 315.24 42.27 140.52 118.09
python 3d_transient_loc.py "GRB160703A" 205243785 287.39 36.90 10.15 95.08
python 3d_transient_loc.py "GRB160821A" 209507850 171.25 42.34 156.18 59.18
python 3d_transient_loc.py "GRB160910A" 211223675 221.8 39.6 65.32 332.32
# For combining the two pdfs (3d plots and the others 
pdftk A=GRB160131A/comparison_GRB160131A.pdf B=GRB160131A/3d_plot_GRB160131A.pdf cat  B A1-end output GRB160131A_vis_test.pdf
pdftk A=GRB160325A/comparison_GRB160325A.pdf B=GRB160325A/3d_plot_GRB160325A.pdf cat  B A1-end output GRB160325A_vis_test.pdf
pdftk A=GRB160509A/comparison_GRB160509A.pdf B=GRB160509A/3d_plot_GRB160509A.pdf cat  B A1-end output GRB160509A_vis_test.pdf
pdftk A=GRB160607A/comparison_GRB160607A.pdf B=GRB160607A/3d_plot_GRB160607A.pdf cat  B A1-end output GRB160607A_vis_test.pdf
pdftk A=GRB160623A/comparison_GRB160623A.pdf B=GRB160623A/3d_plot_GRB160623A.pdf cat  B A1-end output GRB160623A_vis_test.pdf
pdftk A=GRB160703A/comparison_GRB160703A.pdf B=GRB160703A/3d_plot_GRB160703A.pdf cat  B A1-end output GRB160703A_vis_test.pdf
pdftk A=GRB160821A/comparison_GRB160821A.pdf B=GRB160821A/3d_plot_GRB160821A.pdf cat  B A1-end output GRB160821A_vis_test.pdf
pdftk A=GRB160910A/comparison_GRB160910A.pdf B=GRB160910A/3d_plot_GRB160910A.pdf cat  B A1-end output GRB160910A_vis_test.pdf

