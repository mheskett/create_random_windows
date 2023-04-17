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
import seaborn as sns
import argparse

### all files needed
fasta_fai = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa.fai"
blacklist_file = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.nochr.bed"
genome_fasta = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa"
common_snps = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.common.snps.nochr.bed"
repeats_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.repeats.hg19.nochr.bed"
dead_zones_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ncbi.dead.zones.nochr.bed"
problematic_regions_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.problematic.nochr.bed"
whole_genes_file = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.ensemble.coding.whole.genes.bed" ### includes introns and exons of coding genes only
###


def clean_df(df):

    tmp = df.loc[:,[0,1,2,"snps_per_kb","percent_gc","fraction_repeats","fraction_within_coding_genes"]]
    tmp.columns = ["chrom","start","stop","snps_per_kb","percent_gc","fraction_repeats","fraction_within_coding_genes"]

    return tmp.reset_index(drop=True)

def filter_df(df,snps_min=0,snps_max=100,percent_gc_min=0,percent_gc_max=1,fraction_repeats_min=0,
            fraction_repeats_max=1,fraction_within_coding_genes_min=0,fraction_within_coding_genes_max=1):



    return df


def write_df(df):

    df.to_csv(arguments.out_file,sep="\t",index=False,header=True)

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
    sns.kdeplot([float(x) for x in windows_nuc_df[str(num_cols+2)+"_pct_gc"]], clip=(0, 1))
    plt.xlim([0,1])
    plt.suptitle("gc fraction")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.savefig("gc.pdf")

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

    return df # returns ANY bed style DF with an added column for %GC


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
    
    plt.figure()
    sns.kdeplot(window_snps_df["snp_density"], clip=(0, 10),label="snps_per_kb")
    plt.xlim([0,10])
    plt.xticks(list(range(0,11)))
    plt.suptitle("snps_per_kb")
    plt.savefig("snps_per_kb.pdf")

    return windows_snps


def fraction_repeats(windows):
    repeats = pybedtools.BedTool(repeats_file)
    windows_repeats = windows.coverage(repeats)
    windows_repeats_df = windows_repeats.to_dataframe(disable_auto_names=True,header=None)
    last_col=len(windows_repeats_df.columns)-1
    plt.figure()
    sns.kdeplot(windows_repeats_df[last_col], clip=(0, 1),label="fraction_repeats")
    plt.xlim([0,1])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.suptitle("Fraction_Repeat_Sequence")
    plt.savefig("fraction_Repeat_sequence.pdf")

    return windows_repeats


def fraction_coding(windows):
    whole_genes = pybedtools.BedTool(whole_genes_file)
    windows_genes = windows.coverage(whole_genes)
    windows_genes_df = windows_genes.to_dataframe(disable_auto_names=True,header=None)
    
    last_col=len(windows_genes_df.columns)-1

    plt.figure()
    sns.kdeplot(windows_genes_df[last_col], clip=(0, 1),label="fraction_repeats")
    plt.xlim([0,1])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.suptitle("Fraction_whole_gene_Sequence")
    plt.savefig("fraction_genes_sequence.pdf")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make windows")

    parser.add_argument("--bed",
       type=str,
       metavar="[bed file input. no header.]",
       required=False,
       help="")
    parser.add_argument("--out_file",
       type=str,
       metavar="[out file]",
       required=True,
       help="full path to output results")
    parser.add_argument("--make_windows",
       type=str,
       metavar="[make windows or no]",
       required=False,
       help="")
    parser.add_argument("--gc_min",
       type=float,
       metavar="[min fraction of gc]",
       required=False,
       help="")
    parser.add_argument("--gc_max",
       type=float,
       metavar="[max fraction of gc]",
       required=False,
       help="")
    parser.add_argument("--repeats_min",
       type=float,
       metavar="[min fraction of repeats]",
       required=False,
       help="")
    parser.add_argument("--repeats_max",
       type=float,
       metavar="[max fraction of repeats]",
       required=False,
       help="")
    parser.add_argument("--gene_fraction_min",
       type=float,
       metavar="[min fractino of genes in windows allowed]",
       required=False,
       help="")
    parser.add_argument("--gene_fraction_max",
       type=float,
       metavar="[max fraction of genes in windows allowed]",
       required=False,
       help="")
    parser.add_argument("--num_windows",
       type=int,
       metavar="[number of windows wanted]",
       required=False,
       help="")
    parser.add_argument("--length_windows",
       type=int,
       metavar="[length of windows wanted]",
       required=False,
       help="")   

    arguments = parser.parse_args()

    # print(random_windows(50000,5000).to_dataframe(disable_auto_names=True, header=None))
    # print(add_gc(random_windows(50000,5000).to_dataframe(disable_auto_names=True, header=None)))
    write_df(clean_df(add_fraction_coding(add_fraction_repeats(add_gc(add_common_snp_density(remove_blacklist(random_windows(50000,5000)).to_dataframe(disable_auto_names=True, header=None)))))))
    # if arguments.bed:
    #     input_file = pybedtools.BedTool(arguments.bed)
    #     # return all the stats
    #     fraction_coding(input_file)
    #     fraction_repeats(input_file)
    #     common_snp_density(input_file)
    #     calculate_gc(input_file)
    # else:
    #     ## probably need to generate a big pandas DF with all the stats as columns, and then
    #     ## use pandas statements to remove the extremes or select for certain stats
    #     my_windows = remove_blacklist(random_windows(length=arguments.length_windows,number=arguments.num_windows*1.5))
    #     print(my_windows.to_dataframe(disable_auto_names=True, header=None).sample(n=arguments.num_windows))










# common_snp_density(fraction_repeats(calculate_gc(remove_blacklist(random_windows(10000,1000))))) # dont forget random seed)
#common_snp_density(random_windows(20000,500))
#fraction_repeats(random_windows(20000,500))
#fraction_coding(random_windows(50000,5000))

## main program should get stats for a list of windows and then allow you to put in similar stats to make new windows



