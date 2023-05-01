import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pybedtools
import argparse
import seaborn as sns

### all files needed
tss_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ensemble.tss.coding.stranded.final.nochr.unique.bed"
fasta_fai = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa.fai"
blacklist_file = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.nochr.bed"
genome_fasta = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa"
common_snps = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.common.snps.nochr.bed"
repeats_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.repeats.hg19.nochr.bed"
dead_zones_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ncbi.dead.zones.nochr.bed"
problematic_regions_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.problematic.nochr.bed"
whole_genes_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ensemble.coding.whole.genes.bed" ### includes introns and exons of coding genes only
###



def closest_tss(windows):
    ## requires both files to be sorted.
    ## distance to TSS is going to be last column of the new df
    tss=pybedtools.BedTool(tss_file)
    windows_tss = windows.closest(tss,d=True).to_dataframe(disable_auto_names=True, header=None)
    last_col = len(windows_tss.columns)
    plt.figure()
    plt.hist(windows_tss[last_col-1],bins=500)
    plt.xlim([0,2000000])
    # plt.xticks(list(range(0,16)))
    plt.suptitle("Distance to TSS. Mean: "+str(windows_tss[last_col-1].mean()))
    plt.savefig(arguments.bed.rstrip(".bed")+"distance_to_tss.pdf")

    return


def add_tss_distance(df):

    tss = pybedtools.BedTool(tss_file)
    a = pybedtools.BedTool.from_dataframe(df)
    last_col = len(a.columns)
    df_distance = a.closest(tss, d=True).to_dataframe(disable_auto_names=True, header=None)
    df["tss_distance"] = df_distance[last_col-1]

    return df


def clean_df(df):

    tmp = df.loc[:,[0,1,2,"snps_per_kb","percent_gc","fraction_repeats","fraction_within_coding_genes"]]
    tmp.columns = ["chrom","start","stop","snps_per_kb","percent_gc","fraction_repeats","fraction_within_coding_genes"]

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
    plt.figure(figsize=(4,2))
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
    a = pybedtools.BedTool.from_dataframe(df)
    num_cols=len(df.columns)
    # get nuc content, which is added to the second extra column. so get that column and add it to origianl DF.
    tmp = a.nucleotide_content(fi=genome_fasta).to_dataframe(disable_auto_names=True, header=None).drop(0).reset_index(drop=True)
    df["percent_gc"] = tmp[num_cols+1]

    return df


def add_common_snp_density(df):
    
    windows = pybedtools.BedTool.from_dataframe(df)
    num_cols = len(df.columns)
    snps = pybedtools.BedTool(common_snps)
    windows_snps = windows.intersect(snps, c=True)
    window_snps_df = windows_snps.to_dataframe(disable_auto_names=True,header=None)
    df["snps_per_kb"] = window_snps_df[num_cols].astype(int) / ((window_snps_df[2].astype(int) - window_snps_df[1].astype(int)) / 1000)
    
    return df


def add_fraction_repeats(df):
    
    windows = pybedtools.BedTool.from_dataframe(df)
    repeats = pybedtools.BedTool(repeats_file)
    windows_repeats = windows.coverage(repeats)
    windows_repeats_df = windows_repeats.to_dataframe(disable_auto_names=True,header=None)
    last_col = len(windows_repeats_df.columns)-1
    df["fraction_repeats"] = windows_repeats_df[last_col]
    
    return df


def add_fraction_coding(df):
    
    windows = pybedtools.BedTool.from_dataframe(df)
    whole_genes = pybedtools.BedTool(whole_genes_file)
    
    windows_genes = windows.coverage(whole_genes)
    windows_genes_df = windows_genes.to_dataframe(disable_auto_names=True,header=None)
    
    last_col=len(windows_genes_df.columns)-1
    df["fraction_within_coding_genes"] = windows_genes_df[last_col]

    return df

def common_snp_density(windows):
    num_cols = len(windows.to_dataframe().columns)

    snps = pybedtools.BedTool(common_snps)
    windows_snps = windows.intersect(snps,c=True)
    window_snps_df = windows_snps.to_dataframe(disable_auto_names=True,header=None)
    window_snps_df["snp_density"] = window_snps_df[num_cols].astype(int) / ((window_snps_df[2].astype(int) - window_snps_df[1].astype(int)) / 1000)
    
    plt.figure(figsize=(4,2))
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
    plt.figure(figsize=(4,2))
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

    plt.figure(figsize=(4,2))
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
    parser.add_argument("--gc_min",
       type=float,
       metavar="[min fraction of gc]",
       required=False,
       default=0,
       help="min gc fraction")
    parser.add_argument("--gc_max",
       type=float,
       metavar="[max fraction of gc]",
       required=False,
       default=1,
       help="max gc fraction")
    parser.add_argument("--repeats_min",
       type=float,
       metavar="[min fraction of repeats]",
       required=False,
       default=0,
       help="min fraction of repeat derived sequence")
    parser.add_argument("--repeats_max",
       type=float,
       metavar="[max fraction of repeats]",
       required=False,
       default=1,
       help="max fraction of repeat derived sequence")
    parser.add_argument("--snps_per_kb_min",
       type=float,
       metavar="[min fraction of common snps per kb]",
       required=False,
       default=0,
       help="min snps per kb")
    parser.add_argument("--snps_per_kb_max",
       type=float,
       metavar="[max fraction of common snps per kb]",
       required=False,
       default=1000,
       help="max snps per kb")
    parser.add_argument("--gene_fraction_min",
       type=float,
       metavar="[min fractino of genes in windows allowed]",
       required=False,
       default=0,
       help="min fraction of whole gene sequence")
    parser.add_argument("--gene_fraction_max",
       type=float,
       metavar="[max fraction of genes in windows allowed]",
       required=False,
       default=1,
       help="max fraction of whole gene sequence")
    parser.add_argument("--num_windows",
       type=int,
       metavar="[number of windows wanted]",
       required=False,
       help="number of desired windows")
    parser.add_argument("--length_windows",
       type=float,
       metavar="[length of windows wanted]",
       required=False,
       help="length of windows desired")   
    parser.add_argument("--min_tss_distance",
       type=int,
       metavar="[minimum distance to tss]",
       required=False,
       help="minimum distance to tss")  
    parser.add_argument("--max_tss_distance",
       type=int,
       metavar="[maximum distance to tss]",
       required=False,
       help="maximum distance to tss")  
    arguments = parser.parse_args()


    closest_tss(random_windows(50000,10000).sort())
    exit()
    ### use pybeddtools sort with the random windows

    ## MAIN PROG. Do this if no user bed file is provided.
    ##
    ##
    if arguments.bed == None:
        num_windows_to_try = arguments.num_windows*2
        windows_unfiltered = clean_df( 
                                add_tss_distance(
                                add_fraction_coding(
                                add_fraction_repeats(
                                add_gc(
                                add_common_snp_density(
                                remove_blacklist(
                                random_windows(arguments.length_windows,num_windows_to_try)).to_dataframe(disable_auto_names=True, header=None)))))))
        windows_filtered = filter_df(windows_unfiltered,
                                    snps_min=arguments.snps_per_kb_min,
                                    snps_max=arguments.snps_per_kb_max,
                                    percent_gc_min=arguments.gc_min,
                                    percent_gc_max=arguments.gc_max,
                                    fraction_repeats_min=arguments.repeats_min,
                                    fraction_repeats_max=arguments.repeats_max,
                                    fraction_within_coding_genes_min=arguments.gene_fraction_min,
                                    fraction_within_coding_genes_max=arguments.gene_fraction_max)
        
        while len(windows_filtered.index) < arguments.num_windows:

            num_windows_to_try = num_windows_to_try*3

            windows_unfiltered = clean_df(
                                add_tss_distance(
                                add_fraction_coding(
                                add_fraction_repeats(
                                add_gc(
                                add_common_snp_density(
                                remove_blacklist(
                                random_windows(arguments.length_windows,num_windows_to_try)).to_dataframe(disable_auto_names=True, header=None)))))))
            print(windows_unfiltered)
            windows_filtered = filter_df(windows_unfiltered,
                                    snps_min=arguments.snps_per_kb_min,
                                    snps_max=arguments.snps_per_kb_max,
                                    percent_gc_min=arguments.gc_min,
                                    percent_gc_max=arguments.gc_max,
                                    fraction_repeats_min=arguments.repeats_min,
                                    fraction_repeats_max=arguments.repeats_max,
                                    fraction_within_coding_genes_min=arguments.gene_fraction_min,
                                    fraction_within_coding_genes_max=arguments.gene_fraction_max)
            print(windows_filtered)

        final = sample_df(windows_filtered,num=arguments.num_windows)
        write_df(final)
        write_bed(final)

    ### this section for makingm plots of a user given bed file of windows
    if arguments.bed:
        input_file = pybedtools.BedTool(arguments.bed).sort()
        # Make plots. Can add more stats to plots
        fraction_coding(input_file)
        fraction_repeats(input_file)
        common_snp_density(input_file)
        calculate_gc(input_file)
        closest_tss(input_file)

