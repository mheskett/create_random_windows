import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pybedtools
import scipy.stats
import random
import argparse
import seaborn as sns
import statistics

## inputs file A and file B in be format. outputs P-value and maybe some metrics.


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="make windows")
	parser.add_argument("--a_bed",
		type=str,
		metavar="[bed file input. no header.]",
		required=False,
		help="bed file 1")
	parser.add_argument("--b_bed",
		type=str,
		metavar="[bed file input. no header.]",
		required=False,
		help="bed file 2")
	arguments = parser.parse_args()
	

	###
	###
	a_bed = pd.read_csv(arguments.a_bed,sep="\t",header=None)
	b_bed = pd.read_csv(arguments.b_bed,sep="\t",header=None)
	
	a_windows = pybedtools.BedTool(arguments.a_bed)
	b_windows = pybedtools.BedTool(arguments.b_bed)

	real_a_intersect_b = a_windows.intersect(b_windows, wa=True, wb=True).to_dataframe(disable_auto_names=True, header=None)
	
	median_size_a = (a_bed[2] - a_bed[1]).median()
	median_size_b = (b_bed[2] - b_bed[1]).median()

	num_a = len(a_bed)
	num_b = len(b_bed)

	real_total_bases = (a_bed[2]-a_bed[1]).sum() + (b_bed[2]-b_bed[1]).sum()

	real_intersections_per_mb = len(real_a_intersect_b) / real_total_bases * 10**6

	print(real_intersections_per_mb)




	num_iterations=5
	# functionalize and then parallelize this to make the whole thing way faster
	for i in range(num_iterations):

		## parallelize this ezpz
		os.system("python null.window.clustering.py --bed "+arguments.a_bed+" --out_file colocalization_windows_a_tmp_"+str(i))
		
		os.system("python null.window.clustering.py --bed "+arguments.b_bed+" --out_file colocalization_windows_b_tmp_"+str(i))

		print("simulated round "+str(i))

	intersections_per_mb=[]
	for i in range(num_iterations):
		simulated_windows = [pybedtools.BedTool("colocalization_windows_a_tmp_" + str(i) + ".bed"),
							pybedtools.BedTool("colocalization_windows_b_tmp_" + str(i) + ".bed")]
		simulated_dfs = [pybedtools.BedTool("colocalization_windows_a_tmp_" + str(i) + ".bed").to_dataframe(disable_auto_names=True, header=None),
							pybedtools.BedTool("colocalization_windows_b_tmp_" + str(i) + ".bed").to_dataframe(disable_auto_names=True, header=None)]
		a_intersect_b = simulated_windows[0].intersect(simulated_windows[1], wa=True, wb=True).to_dataframe(disable_auto_names=True, header=None)

		num_intersections = len(a_intersect_b)
		total_bases =  (simulated_dfs[0][2]-simulated_dfs[0][1]).sum() + (simulated_dfs[1][2]-simulated_dfs[1][1]).sum()
		intersections_per_mb += [num_intersections / total_bases * 10**6]
		print("intersections per Mb: " + str(num_intersections / total_bases * 10**6))

	sns.kdeplot(intersections_per_mb,cut=0)
	plt.show()



	zscore = statistics.NormalDist(mu = np.array(intersections_per_mb).mean(), 
									sigma = np.array(intersections_per_mb).std()).zscore(real_intersections_per_mb)
	print("z score of real intersections per mb compared to simulated: "+ str(zscore))
	print("mean of intersections per mb fake " + str(np.array(intersections_per_mb).mean()))
	print("std of intersections per mb fake " + str(np.array(intersections_per_mb).std()))
