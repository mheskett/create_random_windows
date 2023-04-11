import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pybedtools
import re
import time
import scipy.stats
import multiprocessing as mp
import random
import json


def min_max(x):
    return min(max(0,x),1)

def is_real_link(x):

    b = pybedtools.BedTool("/Users/heskett/tcga_replication_timing/data/hg19/links.annotated.hg19.bed")
    a = x # returns only non-overlappers with real links
    # x is a bed tool object...

    return a.intersect(b,v=True)

def simulate_links_by_length(length, window_fraction=0.1, minimum=40, maximum=100):

    tcga_file = "/Users/heskett/tcga_replication_timing/data/hg19/ucsc.not.i.e.tcga.covered.bed"

    rand = str(random.randint(0,1000))
    tmp_file = "tmp." + str(length) + "." + rand + "." + "filtered.bed"
    awk_command = " awk '$3-$2>=" + str(length) + "{print $0}' " + tcga_file + " > tmp." + str(length) + "." + rand + "." + "filtered.bed"
    os.system(awk_command)
    
    a=pybedtools.BedTool()
    windows=a.window_maker(b=tmp_file, w=length, s=length*window_fraction)

    #os.system("igv "+str(tmp_file))
    os.system("rm " + tmp_file)
    return len(windows)

def faster_simulate_links(length, window_fraction=0.25, snps=False, l1=False, gc=False, wiggle=0.1, minimum=40, maximum=100):
    """
    returns a list of fake genes, from a length
    predicated on the fact that bedtools.nuc is the only slow step in this analysis
    need to: trim down reference genome to include regions that are not in TCGA
    then: do a check to see how many samples contain each fake gene....
    Need a file that has all non coding regions of genome.
    """
    if length < 20000:
        window_fraction = 1

    start=time.time()
    # snp doesnt need to be bedtool object yet i guess
    snp_file = "/Users/heskett/tcga_replication_timing/data/hg19/ucsc.snp150.nochr.hg19.bed"
    l1_file = pybedtools.BedTool("/Users/heskett/tcga_replication_timing/data/hg19/ucsc.l1.nochr.hg19.bed")

    # initiate bed tool and make windows
    # all file must be sorted sort -k1,1 -k2,2n
    a=pybedtools.BedTool()
    # "/Users/heskett/tcga_replication_timing/data/hg19/TCGA.nochr.bed"
    ### using the covered regions here will result in smaller segments. window maker will just make a window
    ### that covers the whole segment if the segment size is smaller than window size.
    windows=a.window_maker(b="/Users/heskett/tcga_replication_timing/data/hg19/ucsc.not.i.e.tcga.covered.bed",
                            w=length,s=length*window_fraction)
    wint=time.time()
    print(wint-start," window time")

    windows_snp = windows.intersect(snp_file,c=True,sorted=True) # makes one column
    print(windows_snp)
    snpt=time.time()
    print(snpt-wint," snp time")
    #print(str(snpt-gc)+ " time for snp ") ## pre sorted file much faster sort -k1,1 -k2,2n
    
    ## get L1
    windows_snp_l1 = windows_snp.coverage(l1_file)# produces 4 columns
    l1t=time.time()
    print(l1t-snpt," l1 time")


    ## always do nuc on a reduced set of windows
    df = windows_snp_l1.to_dataframe(names=["chrom","start","end",
                                        "num_snps","cov1",
                                        "cov2","cov3","fraction_l1"]) # check that crom is str or int?
    df["snps/kb"] = df["num_snps"] / (length/1000)

    snps_score,l1_score = (scipy.stats.percentileofscore(a=df["snps/kb"], score=snps, kind='rank'), # this giving seg fault?
                            scipy.stats.percentileofscore(a=df["fraction_l1"], score=l1, kind='rank'))
    df = df[
        (df["snps/kb"]\
            .between(left=df["snps/kb"].quantile(q=min_max(snps_score/100-wiggle)), # could separate these for optimization if necessary
                    right=df["snps/kb"].quantile(q=min_max(snps_score/100+wiggle))))\
        & (df["fraction_l1"]\
            .between(left=df["fraction_l1"].quantile(q=min_max(l1_score/100-wiggle)), 
                    right=df["fraction_l1"].quantile(q=min_max(l1_score/100+wiggle))))]
        ## should add another recursive call right here...
    b = pybedtools.BedTool()
    window_snp_l1_reduced = b.from_dataframe(df)
    window_snp_l1_nuc =  window_snp_l1_reduced.nucleotide_content(fi="/Users/heskett/tcga_replication_timing/data/hg19/human_g1k_v37_nochr.fasta")#\
                            #.to_dataframe() ## 0th column has colnames. current colname is garbage

    # columns at this point are...
    # chrom start stop num snps, cov1, cov2, cov3, fraction l1, snps/kb, %AT, %GC, whatever...
    window_snp_l1_nuc = is_real_link(window_snp_l1_nuc)
    window_snp_l1_nuc = window_snp_l1_nuc.to_dataframe()
    window_snp_l1_nuc = window_snp_l1_nuc.loc[:,0:10]
    nuct=time.time()
    print(nuct-snpt," nuc time")
    window_snp_l1_nuc.columns = ["chrom","start","end","num_snps","cov1",
                                "cov2","cov3","fraction_l1","snps/kb","AT","pct_gc"]

    window_snp_l1_nuc[["pct_gc","fraction_l1","snps/kb","start","end"]] = \
    window_snp_l1_nuc[["pct_gc","fraction_l1","snps/kb","start","end"]].apply(pd.to_numeric)
    gc_score = scipy.stats.percentileofscore(window_snp_l1_nuc["pct_gc"], score=gc, kind='rank')
    window_snp_l1_nuc = window_snp_l1_nuc[window_snp_l1_nuc["pct_gc"]\
            .between(left=window_snp_l1_nuc["pct_gc"].quantile(q=min_max(gc_score/100-wiggle)),
                    right=window_snp_l1_nuc["pct_gc"].quantile(q=min_max(gc_score/100+wiggle)))]
    print(len(window_snp_l1_nuc))
    ########
    ## find out how many samples contain each fake link
    #######
    if len(window_snp_l1_nuc) < minimum:
        wiggle+=0.05
        window_snp_l1_nuc = faster_simulate_links(length=length,snps=snps,gc=gc,l1=l1,wiggle=wiggle,minimum=minimum) # recursive biatch
        print(wiggle, " wiggle") # not returning this!!!!
    if len(window_snp_l1_nuc) > maximum:
        window_snp_l1_nuc = window_snp_l1_nuc.sample(n=maximum,axis="rows",random_state=1) # shuld randomly sample rows!!
    final = window_snp_l1_nuc.loc[:,["chrom","start","end",
                    "snps/kb","pct_gc","fraction_l1"]].astype({"chrom":str,"start":int,"end":int,
                        "snps/kb":float,"pct_gc":float,"fraction_l1":float})

    final.loc[:,["chrom","start","end"]].to_csv("simulate.asars."+ str(l1)+ "."+ str(length)+"."+str(snps)+"."+str(gc)+".bed", 
        index=None, header=None, sep="\t")
    print(final)
    return final

def search_tcga(segments_df,link_chromosome,link_start,link_end):
    types = ["gain","loss","neutral","disruption"]
    segments_df = segments_df[segments_df['chr']==link_chromosome] # then remove the additional operation from below...
    #import pdb;pdb.set_trace()
    losses = segments_df[(segments_df['start'] <= link_start) 
                            & (segments_df['stop'] >= link_end) 
                            & (segments_df['copy_number'] <2.0 )]

    gains = segments_df[(segments_df['start'] <= link_start)
                            & (segments_df['stop'] >= link_end) 
                            & (segments_df['copy_number'] > 2.0 )]


    neutrals = segments_df[(segments_df['start'] <= link_start) 
                            & (segments_df['stop'] >= link_end) 
                            & (segments_df['copy_number'] == 2.0 )]
    # max one per sample
    disruptions = segments_df[(segments_df['start'].between(left=link_start, right=link_end)
                                | segments_df['stop'].between(left=link_start, right=link_end))]
    disruptions = disruptions[disruptions.duplicated(subset="patient", keep=False)] # mcaarks all dupes as true cause we want to keep dupes
    disruptions = disruptions.drop_duplicates(subset="patient", keep="first") # keeps the first dupe
    
    losses.loc[:,"type"] = pd.Series(["loss"]*len(losses.index), index=losses.index).astype(str) # empty list was being assigned float64 type...
    gains.loc[:,"type"] = pd.Series(["gain"]*len(gains.index), index=gains.index).astype(str)
    disruptions.loc[:,"type"]= pd.Series(["disruption"]*len(disruptions.index), index=disruptions.index).astype(str)
    neutrals.loc[:,"type"] = pd.Series(["neutral"]*len(neutrals.index), index=neutrals.index).astype(str)
    #print(pd.concat([losses,gains,disruptions,neutrals]))
    return pd.concat([losses,gains,disruptions,neutrals])

#### load data and preprocess

with open ("/Users/heskett/tcga_replication_timing/data/hg19/tcga_cancer_type_dictionary.txt") as f:
    lines = f.readlines()
    lines = (x.rstrip("\n").split("\t") for x in lines)
    cancer_atlas_dictionary = dict(lines)
    f.close()

with open ("/Users/heskett/tcga_replication_timing/data/hg19/TCGA.segtabs.final.merged.sorted.no23.bed") as h:
    segments = h.readlines()
    segments = [x.rstrip("\n").split("\t") for x in segments]
    h.close() # fast


with open ("/Users/heskett/tcga_replication_timing/data/hg19/ucsc.not.i.e.tcga.covered.20kb.min.bed") as i:
    regions = i.readlines()
    regions = [x.rstrip("\n").split("\t") for x in regions]
    regions = [[str(x[0]),int(x[1]),int(x[2])] for x in regions]
    i.close() # fast
segments = [[str(x[0]),
    int(x[1]),
    int(x[2]),
    str(x[3]),
    str(cancer_atlas_dictionary[str(x[3][:-3])]),
    float(x[4])] for x in segments if x[3][:-3] in cancer_atlas_dictionary.keys()] # v slow...2min

# exclude_list = ["ACC","CHOL","DLBC","KICH","MESO","UCS","UVM"]
# segments_df= pd.DataFrame(segments)
# ## get rid of all segments belonging to low coverage tumor types
# segments_df.columns = ["chr","start","stop","patient","cancer_type","copy_number"]
# segments_df = segments_df[~segments_df.cancer_type.isin(exclude_list)]
# ################################
# non_coding_regions = [["nc_region_"+str(x[0])+":"+str(x[1])+"-"+str(x[2]),x[0],x[1],x[2]] for x in regions]

# pool = mp.Pool(7)
# results = pool.starmap(search_tcga,[(segments_df,x[1],x[2],x[3]) for x in non_coding_regions]) ##  each link needs to be list of list in parallel call

# final = pd.DataFrame()
# for i in range(len(results)):
#   results[i]["nc_region_chr"] = non_coding_regions[i][1]
#   results[i]["nc_region_start"] = non_coding_regions[i][2]
#   results[i]["nc_region_stop"] = non_coding_regions[i][3]

# final = pd.concat(results)

# final.to_csv("tcga.search.on.all.noncoding.aug15.txt",sep="\t",header=True,index=None)

# for i in range(len(non_coding_regions)):
#   print("iteration")
#   results[non_coding_regions[i][0]] = search_tcga(segments_df,
#                                           non_coding_regions[i][1],
#                                           non_coding_regions[i][2],
#                                           non_coding_regions[i][3])
# tmp = search_tcga(segments_df,non_coding_regions[0][1],non_coding_regions[0][2],non_coding_regions[0][3])

# tmp["type"].value_counts().plot(kind='bar')
# plt.show()


# tmp.loc[:,["chr","start","stop","type"]].to_csv("test.tcgaserach.bed",sep="\t",index=None,header=None)

# os.system("igv test.tcgaserach.bed")


## this simulates links and then plots in IGV
# files=[]
# for i in range(5):
#   l1=.25 + i*0.05 
#   length=200000
#   gc=0.35
#   snps=3.9
#   faster_simulate_links(length,window_fraction=0.25,snps=snps,l1=l1,gc=gc,wiggle=0.1,minimum=40,maximum=200)
#   files+=["simulate.asars."+ str(l1)+ "."+ str(length)+"."+str(snps)+"."+str(gc)+".bed"]
#   file_string = "igv " + " ".join(files)
# os.system(file_string)


### make a plot of the number of intergenic regions of length N for every length N 100kb to 1MB
print(simulate_links_by_length(200000))
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
tcga_replication_timing/tcga.search.aug19.py at master · mheskett/tcga_replication_timing