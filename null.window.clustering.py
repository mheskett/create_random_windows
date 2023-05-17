import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pybedtools
import argparse
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn.decomposition import PCA

### all files needed
tss_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ensemble.tss.coding.stranded.final.nochr.unique.bed"
fasta_fai = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa.fai"
blacklist_file = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.nochr.bed"
genome_fasta = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa"
common_snps = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.common.snps.nochr.bed"
repeats_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.repeats.hg19.nochr.bed"
dead_zones_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ncbi.dead.zones.nochr.bed"
problematic_regions_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.problematic.nochr.bed"
whole_genes_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ensemble.coding.whole.genes.sorted.bed" ### includes introns and exons of coding genes only
three_utr_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.hg19.3utr.exons.nochr.bed"
five_utr_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.hg19.5utr.exons.nochr.bed"
introns_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.hg19.introns.nochr.sorted.bed"
exons_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.refseq.all.exons.hg19.nochr.bed"
###

### 

## dist AND coverage for UTRs, exons, introns, <---(non specific if coding or noncoding), and then whole coding gene cov and dist


## add distance and coverage
def add_3utr_distance(df):
    
    three_utr = pybedtools.BedTool(three_utr_file)
    a = pybedtools.BedTool.from_dataframe(df)
    df_distance = a.closest(three_utr, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)

    df["three_utr_distance"] = df_distance[df_distance.columns[-1]].tolist()

    windows_utrs = a.coverage(three_utr)
    windows_utrs_df = windows_utrs.to_dataframe(disable_auto_names=True,header=None)
    df["fraction_three_utr"] = windows_utrs_df[windows_utrs_df.columns[-1]].tolist()

    return df

def add_5utr_distance(df):
    
    five_utr = pybedtools.BedTool(five_utr_file)
    a = pybedtools.BedTool.from_dataframe(df.loc[:,[0,1,2]])
    df_distance = a.closest(five_utr, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df["five_utr_distance"] = df_distance[df_distance.columns[-1]].tolist()

    windows_utrs = a.coverage(five_utr)
    windows_utrs_df = windows_utrs.to_dataframe(disable_auto_names=True,header=None)
    df["fraction_five_utr"] = windows_utrs_df[windows_utrs_df.columns[-1]].tolist()
    
    
    return df


def add_intron_distance(df):
    
    introns = pybedtools.BedTool(introns_file)
    a = pybedtools.BedTool.from_dataframe(df)
    df_distance = a.closest(introns, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df["intron_distance"] = df_distance[df_distance.columns[-1]].tolist()

    windows_introns = a.coverage(introns)
    windows_introns_df = windows_introns.to_dataframe(disable_auto_names=True,header=None)
    df["fraction_introns"] = windows_introns_df[windows_introns_df.columns[-1]].tolist()
    
    return df

def add_exon_distance(df):
    
    exons = pybedtools.BedTool(exons_file)
    a = pybedtools.BedTool.from_dataframe(df)
    df_distance = a.closest(exons, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df["exon_distance"] = df_distance[df_distance.columns[-1]].tolist()

    windows_exons = a.coverage(exons)
    windows_exons_df = windows_exons.to_dataframe(disable_auto_names=True,header=None)
    df["fraction_exons"] = windows_exons_df[windows_exons_df.columns[-1]].tolist()
    
    return df

def add_whole_coding_gene_distance(df):
    
    whole_gene = pybedtools.BedTool(whole_genes_file)
    a = pybedtools.BedTool.from_dataframe(df)
    df_distance = a.closest(whole_gene, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df["whole_coding_gene_distance"] = df_distance[df_distance.columns[-1]].tolist()

    windows_whole_gene = a.coverage(whole_gene)
    windows_whole_gene_df = windows_whole_gene.to_dataframe(disable_auto_names=True,header=None)
    df["fraction_whole_coding_gene_distance"] = windows_whole_gene_df[windows_whole_gene_df.columns[-1]].tolist()
    
    return df



def remove_reals(df_sims, df_reals):

    reals = pybedtools.BedTool.from_dataframe(df_reals)
    sims = pybedtools.BedTool.from_dataframe(df_sims)
    tmp = sims.closest(reals, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df_sims["dist_to_real"]=tmp[tmp.columns[-1]].tolist()
    return df_sims[df_sims["dist_to_real"]>0].drop("dist_to_real",axis=1).reset_index(drop=True)


def plot_lengths(windows):
    df = pybedtools.BedTool(windows).to_dataframe(disable_auto_names=True, header=None)

    plt.figure()
    lengths = (df[2]-df[1])
    plt.hist(lengths,bins=100)
    # plt.xlim([0,2000000]
    # plt.xticks(list(range(0,16)))
    plt.suptitle("window length. Mean: "+str(lengths.mean()))
    plt.savefig(arguments.bed.rstrip(".bed")+"lengths.pdf")

    return


def closest_tss(windows):

    ## not going to work when you have weird chromosome names like "gl_"
    ## requires both files to be sorted.
    ## distance to TSS is going to be last column of the new df
    tss=pybedtools.BedTool(tss_file)
    windows_tss = windows.closest(tss,d=True).to_dataframe(disable_auto_names=True, header=None)
    last_col = len(windows_tss.columns)
    plt.figure()
    plt.hist(np.log2(windows_tss[last_col-1]+1),bins=100)
    # plt.xlim([0,4])
    # plt.xticks(list(range(0,5)))
    plt.suptitle("Distance to TSS. Mean: "+str(windows_tss[last_col-1].mean()))
    plt.savefig(arguments.bed.rstrip(".bed")+"distance_to_tss.pdf")

    return


def add_tss_distance(df):

    tss = pybedtools.BedTool(tss_file)
    a = pybedtools.BedTool.from_dataframe(df)
    ### sorted without sorting input. thats a serious bug!!!
    df_distance = a.closest(tss, d=True, t="first").to_dataframe(disable_auto_names=True, header=None)
    df["tss_distance"] = df_distance[df_distance.columns[-1]]

    return df


def clean_df(df):

    tmp = df.loc[:,[0,1,2,"snps_per_kb","percent_gc","fraction_repeats",
                            "three_utr_distance","fraction_three_utr","five_utr_distance","fraction_five_utr",
                            "whole_coding_gene_distance","fraction_whole_coding_gene_distance","tss_distance",
                            "intron_distance","fraction_introns","exon_distance","fraction_exons"]]

    tmp.columns = ["chrom","start","stop","snps_per_kb","percent_gc","fraction_repeats",
                            "three_utr_distance","fraction_three_utr","five_utr_distance","fraction_five_utr",
                            "whole_coding_gene_distance","fraction_whole_coding_gene_distance","tss_distance",
                            "intron_distance","fraction_introns","exon_distance","fraction_exons"]

    tmp = tmp[tmp["chrom"]!="Y"]

    return tmp.reset_index(drop=True)

def filter_df(df, snps_min=0,
                snps_max=100,
                percent_gc_min=0,
                percent_gc_max=1,
                fraction_repeats_min=0,
                fraction_repeats_max=1,
                fraction_within_coding_genes_min=0,
                fraction_within_coding_genes_max=1,
                min_tss_dist=0,
                max_tss_dist=3*10**9):

    if snps_min > snps_max:
        print("error in snps argument")
        return
    if percent_gc_min > percent_gc_max:
        print("error in gc argument")
        return
    if fraction_repeats_min > fraction_repeats_max:
        print("error in repeats argument")
        return
    if fraction_within_coding_genes_min > fraction_within_coding_genes_max:
        print("error in coding genes argument")
        return

    print("before filtering")
    print(df)
    tmp = df[(df["snps_per_kb"].astype(float) >= snps_min) & 
            (df["snps_per_kb"].astype(float) <= snps_max) & 
            (df["percent_gc"].astype(float) >= percent_gc_min) & 
            (df["percent_gc"].astype(float) <= percent_gc_max) &
            (df["fraction_repeats"].astype(float) >= fraction_repeats_min) &
            (df["fraction_repeats"].astype(float) <= fraction_repeats_max) &
            (df["fraction_within_coding_genes"].astype(float) >= fraction_within_coding_genes_min) & 
            (df["fraction_within_coding_genes"].astype(float) <= fraction_within_coding_genes_max) &
            (df["tss_distance"].astype(float) >= min_tss_dist) &
            (df["tss_distance"].astype(float) <= max_tss_dist)]

    print("after filtering")
    print(tmp)

    return tmp

def sample_df(df,num):

    if num >= len(df.index):
        print("ERROR: asking for fewer rows than exist in the data frame")
        return 

    return df.sample(n=num)


def write_df(df):

    df.to_csv(arguments.out_file,sep="\t",index=False,header=True)

    return

def write_bed(df):

    df.loc[:,["chrom","start","stop"]].to_csv(arguments.out_file.rstrip(".txt")+'.bed',sep="\t",index=False,header=False)

    return

def random_windows(length,number, write_file=False, file_path=None):
    a=pybedtools.BedTool()
    windows = a.random(l=length, n=number, g=fasta_fai)
    ## this returns (chr start stop number index, length, random strand.)

    return windows


def remove_blacklist(windows):
    blacklist = pybedtools.BedTool(blacklist_file)
    problematic_regions = pybedtools.BedTool(problematic_regions_file)
    dead_zones = pybedtools.BedTool(dead_zones_file)
    windows = windows.subtract(blacklist, A=True).subtract(problematic_regions,A=True).subtract(dead_zones,A=True)

    return windows


def calculate_gc(windows):
    # can either chop at chrom start stop, or keep everything and then create the header basd on how long
    # names=["#1_usercol", "2_usercol", "3_usercol",  "4_pct_at",  "5_pct_gc",  "6_num_A",    "7_num_C",  "8_num_G",  "9_num_T",   "10_num_N",  "11_num_oth",  "12_seq_len"],
    num_cols = len(windows.to_dataframe().columns)
    windows_nuc = windows.nucleotide_content(fi=genome_fasta)
    windows_nuc_df = windows_nuc.to_dataframe(disable_auto_names=True)
    plt.figure()
    dat = [float(x) for x in windows_nuc_df[str(num_cols+2)+"_pct_gc"]]
    plt.hist(dat, bins=50)
    plt.xlim([0,1])
    plt.suptitle("gc fraction. Mean: "+str(np.array(dat).mean()))
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.savefig(arguments.bed.rstrip(".bed")+"_gc.pdf")

    return windows_nuc


def add_gc(df):
    # start and end with pandas DFs
    """
    Output format: 
            The following information will be reported after each BED entry:
                1) %AT content
                2) %GC content
                3) Number of As observed
                4) Number of Cs observed
                5) Number of Gs observed
                6) Number of Ts observed
                7) Number of Ns observed
                8) Number of other bases observed
                9) The length of the explored sequence/interval.
                10) The seq. extracted from the FASTA file. (opt., if -seq is used)
                11) The number of times a user's pattern was observed.
                    (opt., if -pattern is used.)
    """
    a = pybedtools.BedTool.from_dataframe(df.reset_index(drop=True))
    # get nuc content, which is added to the second extra column. so get that column and add it to origianl DF.
    tmp = a.nucleotide_content(fi=genome_fasta).to_dataframe(disable_auto_names=True, header=None).drop(0).reset_index(drop=True)
    df["percent_gc"] = tmp[tmp.columns[-9]].tolist()

    return df


def add_common_snp_density(df):
    
    windows = pybedtools.BedTool.from_dataframe(df.reset_index(drop=True))
    snps = pybedtools.BedTool(common_snps)
    windows_snps = windows.intersect(snps, wa=True, c=True) # wa true seems to fix, but honestly i think its a legit bug.....
    window_snps_df = windows_snps.to_dataframe(disable_auto_names=True,header=None) ## somehow shortr than the regular df
    calc = (window_snps_df[window_snps_df.columns[-1]].astype(int) / ((window_snps_df[2].astype(int) - window_snps_df[1].astype(int)) / 1000)).tolist()
    calc = [round(x,1) for x in calc]
    df["snps_per_kb"] = calc
    # df["snps"] = window_snps_df[window_snps_df.columns[-1]].astype(int).tolist()

    return df


def add_fraction_repeats(df):
    
    windows = pybedtools.BedTool.from_dataframe(df)
    repeats = pybedtools.BedTool(repeats_file)
    windows_repeats = windows.coverage(repeats)
    windows_repeats_df = windows_repeats.to_dataframe(disable_auto_names=True,header=None)
    last_col = len(windows_repeats_df.columns)-1
    df["fraction_repeats"] = windows_repeats_df[last_col]
    
    return df


# def add_fraction_coding(df):
    
#     windows = pybedtools.BedTool.from_dataframe(df)
#     whole_genes = pybedtools.BedTool(whole_genes_file)
    
#     windows_genes = windows.coverage(whole_genes)
#     windows_genes_df = windows_genes.to_dataframe(disable_auto_names=True,header=None)
    
#     last_col=len(windows_genes_df.columns)-1
#     df["fraction_within_coding_genes"] = windows_genes_df[last_col]

#     # print("windows genes df")
#     # print(windows_genes_df)
#     # windows_genes_df.to_csv("test234.txt",sep="\t")
#     # print("df")
#     # print(df)

#     return df

def common_snp_density(windows):
    num_cols = len(windows.to_dataframe().columns)

    snps = pybedtools.BedTool(common_snps)
    windows_snps = windows.intersect(snps,c=True)
    window_snps_df = windows_snps.to_dataframe(disable_auto_names=True,header=None)
    window_snps_df["snp_density"] = window_snps_df[num_cols].astype(int) / ((window_snps_df[2].astype(int) - window_snps_df[1].astype(int)) / 1000)
    
    plt.figure()
    plt.hist(window_snps_df["snp_density"],bins=500)
    plt.xlim([0,15])
    plt.xticks(list(range(0,16)))
    plt.suptitle("snps_per_kb. Mean: "+str(window_snps_df["snp_density"].mean()))
    plt.savefig(arguments.bed.rstrip(".bed")+"_snps_per_kb.pdf")

    return windows_snps


def fraction_repeats(windows):
    repeats = pybedtools.BedTool(repeats_file)
    windows_repeats = windows.coverage(repeats)
    windows_repeats_df = windows_repeats.to_dataframe(disable_auto_names=True,header=None)
    last_col=len(windows_repeats_df.columns)-1
    plt.figure()
    plt.hist(windows_repeats_df[last_col],bins=50)
    plt.xlim([0,1])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.suptitle("Fraction_Repeat_Sequence. Mean: "+str(windows_repeats_df[last_col].mean()))
    plt.savefig(arguments.bed.rstrip(".bed")+"_fraction_Repeat_sequence.pdf")

    return windows_repeats


def fraction_coding(windows):
    whole_genes = pybedtools.BedTool(whole_genes_file)
    windows_genes = windows.coverage(whole_genes)
    windows_genes_df = windows_genes.to_dataframe(disable_auto_names=True,header=None)
    
    last_col=len(windows_genes_df.columns)-1
    plt.figure()
    # sns.kdeplot(windows_genes_df[last_col], clip=(0, 1),label="fraction_repeats")
    plt.xlim([0,1])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.suptitle("Fraction_whole_gene_Sequence. Mean: "+str(windows_genes_df[last_col].mean()))
    plt.hist(windows_genes_df[last_col],bins=50)
    plt.savefig(arguments.bed.rstrip(".bed")+"_fraction_genes_sequence.pdf")


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make windows")

    parser.add_argument("--bed",
       type=str,
       metavar="[bed file input. no header.]",
       required=False,
       help="input bed file to plot genome features distributions")
    parser.add_argument("--out_file",
       type=str,
       metavar="[out file]",
       required=True,
       help="full path to output results")
    parser.add_argument("--make_plots",
       type=str,
       metavar="[to make plots or not]",
       required=False,
       help="outputting plots or not")

    arguments = parser.parse_args()



# df = pd.read_csv("GSM3563751_mcf7_wt_e2_peaks_liftover_hg19.nochr.filtered.bed",sep="\t")
# windows_file = pybedtools.BedTool("GSM3563751_mcf7_wt_e2_peaks_liftover_hg19.nochr.filtered.bed")

windows_file = pybedtools.BedTool(arguments.bed)
windows_file_df = pd.read_csv(arguments.bed,sep="\t", header=None)
median_length = (windows_file_df[2] - windows_file_df[1]).median()
number = len(windows_file_df)

# windows_file = pybedtools.BedTool("amplicon_segments_project7_ecDNA_positive.bed")
windows = clean_df(
            add_exon_distance(
            add_tss_distance(
            add_intron_distance(
            add_whole_coding_gene_distance(
            add_5utr_distance(
            add_3utr_distance(
            add_fraction_repeats(
            add_gc(
            add_common_snp_density( # add common snp return NANs
            remove_blacklist(windows_file.sort()).to_dataframe(disable_auto_names=True, header=None)))))))))))

# print("windows",windows)
# print('debug',add_gc(add_common_snp_density(remove_reals(df_sims=remove_blacklist(random_windows(470,100000).sort()).to_dataframe(disable_auto_names=True, header=None),
#                         df_reals=windows))))

simulated_windows = clean_df( 
            add_exon_distance(
            add_tss_distance(
            add_intron_distance(
            add_whole_coding_gene_distance(
            add_5utr_distance(
            add_3utr_distance(
            add_fraction_repeats(
            add_gc(
            add_common_snp_density(
            remove_reals(df_sims=remove_blacklist(random_windows(median_length,number*5).sort()).to_dataframe(disable_auto_names=True, header=None),
                        df_reals=windows).reset_index(drop=True)))))))))))



##############
combined = pd.concat([windows,simulated_windows])
###

combined["three_utr_distance"] = np.log2(combined["three_utr_distance"]+1)
combined["five_utr_distance"] = np.log2(combined["five_utr_distance"]+1)
combined["whole_coding_gene_distance"] = np.log2(combined["whole_coding_gene_distance"]+1)
combined["tss_distance"] = np.log2(combined["tss_distance"]+1)
combined["exon_distance"] = np.log2(combined["exon_distance"]+1)
combined["intron_distance"] = np.log2(combined["intron_distance"]+1)


# combined.to_csv("testdf9.txt",sep="\t")

## try scaling together
combined_scaled = preprocessing.scale(combined.loc[:,["snps_per_kb","percent_gc","fraction_repeats",
                            "three_utr_distance","fraction_three_utr","five_utr_distance","fraction_five_utr",
                            "whole_coding_gene_distance","fraction_whole_coding_gene_distance","tss_distance",
                            "intron_distance","fraction_introns","exon_distance","fraction_exons"]].reset_index(drop=True))
dist_mat = sklearn.metrics.pairwise.euclidean_distances(X=combined_scaled[0:len(windows),:],Y=combined_scaled[len(windows):,:])


# get indices of nearest euclidean neighbors.
indices = []
## slow algorithm
for i in range(len(dist_mat)):
    closest = np.argmin(dist_mat[i])
    if closest not in indices:
        indices += [closest]
        # print("added first closest")
    else:
        index=1
        tmp_list=list(dist_mat[i])
        tmp_sorted = sorted(dist_mat[i])
        while closest in indices:
            closest = tmp_list.index(tmp_sorted[index]) # this can go out of range if too few sim windows 
            index += 1
        indices += [closest]

# get index of Y with lowest distance
indices_added = [x+len(windows) for  x in indices]
simulated_windows.loc[indices,:].to_csv(arguments.out_file,sep="\t",header=None,index=None)


if arguments.make_plots:
#####
    tsne=TSNE(n_components=2)
    dat_tsne_test = tsne.fit_transform(combined_scaled)


    #####
    fig,ax = plt.subplots(1,3)
    ax[0].scatter(dat_tsne_test[:,0],dat_tsne_test[:,1],s=20,lw=0.5,edgecolor="black",c="blue")
    ax[1].scatter(dat_tsne_test[indices_added,0],dat_tsne_test[indices_added,1],s=20,lw=0.5,edgecolor="black",c="blue")
    ax[2].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c="red")

    ax[0].set_xlim([-80,80])
    ax[0].set_ylim([-80,80])
    ax[1].set_xlim([-80,80])
    ax[1].set_ylim([-80,80])
    ax[2].set_xlim([-80,80])
    ax[2].set_ylim([-80,80])
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_title("All simulated binding sites")
    ax[1].set_title("Nearest simulated Neighbor to each real binding sites")
    ax[2].set_title("Real binding sites")


    plt.show()
    plt.close()
    ######


    ###
    ## TSNE TSNE TSNE

    fig,ax=plt.subplots(2,14)
    ax[0,0].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,0],cmap="Blues")
    ax[0,1].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,1],cmap="Blues")
    ax[0,2].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,2],cmap="Blues")
    ax[0,3].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,3],cmap="Blues")
    ax[0,4].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,4],cmap="Blues")
    ax[0,5].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,5],cmap="Blues")
    ax[0,6].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,6],cmap="Blues")
    ax[0,7].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,7],cmap="Blues")
    ax[0,8].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,8],cmap="Blues")
    ax[0,9].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,9],cmap="Blues")
    ax[0,10].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,10],cmap="Blues")
    ax[0,11].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,11],cmap="Blues")
    ax[0,12].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,12],cmap="Blues")
    ax[0,13].scatter(dat_tsne_test[len(windows):,0],dat_tsne_test[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[len(windows):,13],cmap="Blues")



    ax[1,0].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),0],cmap="Reds")
    ax[1,1].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),1],cmap="Reds")
    ax[1,2].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),2],cmap="Reds")
    ax[1,3].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),3],cmap="Reds")
    ax[1,4].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),4],cmap="Reds")
    ax[1,5].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),5],cmap="Reds")
    ax[1,6].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),6],cmap="Reds")
    ax[1,7].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),7],cmap="Reds")
    ax[1,8].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),8],cmap="Reds")
    ax[1,9].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),9],cmap="Reds")
    ax[1,10].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),10],cmap="Reds")
    ax[1,11].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),11],cmap="Reds")
    ax[1,12].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),12],cmap="Reds")
    ax[1,13].scatter(dat_tsne_test[0:len(windows),0],dat_tsne_test[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=combined_scaled[0:len(windows),13],cmap="Reds")



    ax[0,0].set_title("sim snps per kb",fontdict={"fontsize":8})
    ax[0,1].set_title("sim percent gc",fontdict={"fontsize":8})
    ax[0,2].set_title("sim frac repeats",fontdict={"fontsize":8})
    ax[0,3].set_title("sim 3UTR dist",fontdict={"fontsize":8})
    ax[0,4].set_title("sim frac 3UTR",fontdict={"fontsize":8})
    ax[0,5].set_title("sim 5UTR dist",fontdict={"fontsize":8})
    ax[0,6].set_title("sim frac 5UTR",fontdict={"fontsize":8})
    ax[0,7].set_title("sim coding gene dist",fontdict={"fontsize":8})
    ax[0,8].set_title("sim frac coding gene",fontdict={"fontsize":8})
    ax[0,9].set_title("sim tss dist",fontdict={"fontsize":8})
    ax[0,10].set_title("sim intron dist",fontdict={"fontsize":8})
    ax[0,11].set_title("sim frac intron",fontdict={"fontsize":8})
    ax[0,12].set_title("sim exon dist",fontdict={"fontsize":8})
    ax[0,13].set_title("sim frac exon",fontdict={"fontsize":8})


    ax[1,0].set_title("snps per kb",fontdict={"fontsize":8})
    ax[1,1].set_title("percent gc",fontdict={"fontsize":8})
    ax[1,2].set_title("frac repeats",fontdict={"fontsize":8})
    ax[1,3].set_title("3UTR dist",fontdict={"fontsize":8})
    ax[1,4].set_title("frac 3UTR",fontdict={"fontsize":8})
    ax[1,5].set_title("5UTR dist",fontdict={"fontsize":8})
    ax[1,6].set_title("frac 5UTR",fontdict={"fontsize":8})
    ax[1,7].set_title("coding gene dist",fontdict={"fontsize":8})
    ax[1,8].set_title("frac coding gene",fontdict={"fontsize":8})
    ax[1,9].set_title("tss dist",fontdict={"fontsize":8})
    ax[1,10].set_title("intron dist",fontdict={"fontsize":8})
    ax[1,11].set_title("frac intron",fontdict={"fontsize":8})
    ax[1,12].set_title("exon dist",fontdict={"fontsize":8})
    ax[1,13].set_title("frac exon",fontdict={"fontsize":8})


    # ax[0,0].set_xlim([-5,5])
    # ax[0,1].set_xlim([-5,5])
    # ax[0,2].set_xlim([-5,5])
    # ax[0,3].set_xlim([-5,5])
    # ax[0,4].set_xlim([-5,5])
    # ax[1,0].set_xlim([-5,5])
    # ax[1,1].set_xlim([-5,5])
    # ax[1,2].set_xlim([-5,5])
    # ax[1,3].set_xlim([-5,5])
    # ax[1,4].set_xlim([-5,5])

    # ax[0,0].set_ylim([-3,5])
    # ax[0,1].set_ylim([-3,5])
    # ax[0,2].set_ylim([-3,5])
    # ax[0,3].set_ylim([-3,5])
    # ax[0,4].set_ylim([-3,5])
    # ax[1,0].set_ylim([-3,5])
    # ax[1,1].set_ylim([-3,5])
    # ax[1,2].set_ylim([-3,5])
    # ax[1,3].set_ylim([-3,5])
    # ax[1,4].set_ylim([-3,5])

    ax[0,0].grid()
    ax[0,1].grid()
    ax[0,2].grid()
    ax[0,3].grid()
    ax[0,4].grid()
    ax[0,5].grid()
    ax[0,6].grid()
    ax[0,7].grid()
    ax[0,8].grid()
    ax[0,9].grid()
    ax[0,10].grid()
    ax[0,11].grid()

    ax[1,0].grid()
    ax[1,1].grid()
    ax[1,2].grid()
    ax[1,3].grid()
    ax[1,4].grid()
    ax[1,5].grid()
    ax[1,6].grid()
    ax[1,7].grid()
    ax[1,8].grid()
    ax[1,9].grid()
    ax[1,10].grid()
    ax[1,11].grid()


    plt.show()


exit()
####
#PCA PCA PCA

fig,ax=plt.subplots(2,5)
ax[0,0].scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=scaled[len(windows):,0],cmap="Blues")
ax[0,1].scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=scaled[len(windows):,1],cmap="Blues")
ax[0,2].scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=scaled[len(windows):,2],cmap="Blues")
ax[0,3].scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=scaled[len(windows):,3],cmap="Blues")
ax[0,4].scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c=scaled[len(windows):,4],cmap="Blues")

ax[1,0].scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=scaled[0:len(windows),0],cmap="Reds")
ax[1,1].scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=scaled[0:len(windows),1],cmap="Reds")
ax[1,2].scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=scaled[0:len(windows),2],cmap="Reds")
ax[1,3].scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=scaled[0:len(windows),3],cmap="Reds")
ax[1,4].scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c=scaled[0:len(windows),4],cmap="Reds")

ax[0,0].set_title("simulated snps per kb",fontdict={"fontsize":8})
ax[0,1].set_title("simulated percent gc",fontdict={"fontsize":8})
ax[0,2].set_title("simulated fraction repeats",fontdict={"fontsize":8})
ax[0,3].set_title("simulated fraction within coding genes",fontdict={"fontsize":8})
ax[0,4].set_title("simulated distance to TSS",fontdict={"fontsize":8})
ax[1,0].set_title("real binding sites snps per kb",fontdict={"fontsize":8})
ax[1,1].set_title("real binding sites percent gc",fontdict={"fontsize":8})
ax[1,2].set_title("real binding sites fraction repeats",fontdict={"fontsize":8})
ax[1,3].set_title("real binding sites fraction within coding genes",fontdict={"fontsize":8})
ax[1,4].set_title("real binding sites distance to TSS",fontdict={"fontsize":8})


ax[0,0].set_xlim([-5,5])
ax[0,1].set_xlim([-5,5])
ax[0,2].set_xlim([-5,5])
ax[0,3].set_xlim([-5,5])
ax[0,4].set_xlim([-5,5])
ax[1,0].set_xlim([-5,5])
ax[1,1].set_xlim([-5,5])
ax[1,2].set_xlim([-5,5])
ax[1,3].set_xlim([-5,5])
ax[1,4].set_xlim([-5,5])

ax[0,0].set_ylim([-3,5])
ax[0,1].set_ylim([-3,5])
ax[0,2].set_ylim([-3,5])
ax[0,3].set_ylim([-3,5])
ax[0,4].set_ylim([-3,5])
ax[1,0].set_ylim([-3,5])
ax[1,1].set_ylim([-3,5])
ax[1,2].set_ylim([-3,5])
ax[1,3].set_ylim([-3,5])
ax[1,4].set_ylim([-3,5])

ax[0,0].grid()
ax[0,1].grid()
ax[0,2].grid()
ax[0,3].grid()
ax[0,4].grid()
ax[1,0].grid()
ax[1,1].grid()
ax[1,2].grid()
ax[1,3].grid()
ax[1,4].grid()


plt.show()

# plt.scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c="red")
# plt.scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c="blue",alpha=0.2)
# plt.show()

# plt.scatter(dat[0:len(windows),0],dat[0:len(windows),1],s=20,lw=0.5,edgecolor="black",c="red",alpha=0.3)
# plt.scatter(dat[len(windows):,0],dat[len(windows):,1],s=20,lw=0.5,edgecolor="black",c="blue")
# plt.show()


exit()



    ### use pybeddtools sort with the random windows

    ## MAIN PROG. Do this if no user bed file is provided.
    ##
    ##
#     if arguments.bed == None:
#         num_windows_to_try = arguments.num_windows*2
#         windows_unfiltered = clean_df( 
#                                 add_tss_distance(
#                                 add_fraction_coding(
#                                 add_fraction_repeats(
#                                 add_gc(
#                                 add_common_snp_density(
#                                 remove_blacklist(
#                                 random_windows(arguments.length_windows,num_windows_to_try).sort()).to_dataframe(disable_auto_names=True, header=None)))))))
#         windows_filtered = filter_df(windows_unfiltered,
#                                     snps_min=arguments.snps_per_kb_min,
#                                     snps_max=arguments.snps_per_kb_max,
#                                     percent_gc_min=arguments.gc_min,
#                                     percent_gc_max=arguments.gc_max,
#                                     fraction_repeats_min=arguments.repeats_min,
#                                     fraction_repeats_max=arguments.repeats_max,
#                                     fraction_within_coding_genes_min=arguments.gene_fraction_min,
#                                     fraction_within_coding_genes_max=arguments.gene_fraction_max,
#                                     min_tss_dist=arguments.min_tss_distance,
#                                     max_tss_dist=arguments.max_tss_distance)
        
#         while len(windows_filtered.index) < arguments.num_windows:

#             num_windows_to_try = num_windows_to_try*3

#             windows_unfiltered = clean_df(
#                                 add_tss_distance(
#                                 add_fraction_coding(
#                                 add_fraction_repeats(
#                                 add_gc(
#                                 add_common_snp_density(
#                                 remove_blacklist(
#                                 random_windows(arguments.length_windows,num_windows_to_try).sort()).to_dataframe(disable_auto_names=True, header=None)))))))
#             print(windows_unfiltered)
#             windows_filtered = filter_df(windows_unfiltered,
#                                     snps_min=arguments.snps_per_kb_min,
#                                     snps_max=arguments.snps_per_kb_max,
#                                     percent_gc_min=arguments.gc_min,
#                                     percent_gc_max=arguments.gc_max,
#                                     fraction_repeats_min=arguments.repeats_min,
#                                     fraction_repeats_max=arguments.repeats_max,
#                                     fraction_within_coding_genes_min=arguments.gene_fraction_min,
#                                     fraction_within_coding_genes_max=arguments.gene_fraction_max,
#                                     min_tss_dist=arguments.min_tss_distance,
#                                     max_tss_dist=arguments.max_tss_distance)
#             print(windows_filtered)

#         final = sample_df(windows_filtered,num=arguments.num_windows)
#         write_df(final)
#         write_bed(final)
#         print("wrote simulated windows")

#     ### this section for makingm plots of a user given bed file of windows
#     if arguments.bed:
#         input_file = pybedtools.BedTool(arguments.bed).sort()
#         # Make plots. Can add more stats to plots
#         fraction_coding(input_file)
#         fraction_repeats(input_file)
#         common_snp_density(input_file)
#         calculate_gc(input_file)
#         closest_tss(input_file)
#         plot_lengths(input_file)

# ### new idea
### make a lot of fake windows, then do K means or dbscan clustering and visualize with t-sne or umap. compare
### where these are on the plot with the real windows.


