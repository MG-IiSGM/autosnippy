#!/usr/bin/env python

import os
import argparse

# ARGUMENTS


def get_arguments():

    parser = argparse.ArgumentParser(
        prog='autosnippy.py', description='Pipeline to call variants (SNVs) with any non model haploid organism using snippy')

    input_group = parser.add_argument_group('Input', 'Input parameters')

    input_group.add_argument('-i', '--input', dest="input_dir", metavar="input_directory",
                             type=str, required=True, help='REQUIRED.Input directory containing all fast[aq] files')
    input_group.add_argument('-o', '--output', type=str, required=True,
                              help='REQUIRED. Output directory to extract all results')
    input_group.add_argument('-r', '--reference', metavar="reference",
                             type=str, required=True, help='REQUIRED. File to map against')
    input_group.add_argument('-s', '--sample', metavar="sample", type=str,
                             required=False, help='Sample to identify further files')
    input_group.add_argument('-L', '--sample_list', type=str, required=False,
                             help='Sample names to analyse only in the file supplied')
    input_group.add_argument('-T', '--threads', type=str, dest="threads",
                              required=False, default=32, help='Threads to use')


    quality_group = parser.add_argument_group(
        'Quality parameters', 'parameters for diferent triming conditions')

    quality_group.add_argument('-c', '--coverage20', type=int, default=70, required=False,
                               help='Minimum percentage of coverage at 20x to clasify as uncovered (Default 70)')
    quality_group.add_argument('-u', '--unmapped', type=int, default=25, required=False,
                               help='Maximum percentage of unmapped reads to classify as uncovered Default: 25')
    quality_group.add_argument('-n', '--min_snp', type=int, required=False,
                               default=30, help='SNP number to pass quality threshold')


    annot_group = parser.add_argument_group(
        'Annotation', 'parameters for variant annotation')

    annot_group.add_argument('-B', '--annot_bed', type=str, default=[],
                             required=False, action='append', help='BED file to annotate')
    annot_group.add_argument('-V', '--annot_vcf', type=str, default=[],
                             required=False, action='append', help='VCF file to annotate')
    annot_group.add_argument('-A', '--annot_aa', type=str, default=[],
                             required=False, action='append', help='aminoacid file to annotate')
    annot_group.add_argument('--mash_database', dest="mash_db", type=str, required=False,
                             default=False, help='MASH ncbi annotation containing species database')
    annot_group.add_argument("--kraken2", dest="kraken2_db", type=str, default=False,
                             required=False, help="Kraken2 database")
    annot_group.add_argument('--snpeff_database', type=str, required=False,
                             default=False, help='snpEFF annotation database')


    compare_group = parser.add_argument_group(
        'Compare', 'parameters for compare_snp')

    compare_group.add_argument('--core', required=False, action='store_true', help='Run snippy-core')
    compare_group.add_argument('-w', '--window', required=False,
                        type=int, default=10, help='Range of bases for variant accumulation to be considered as a hotspot: default 10')
    compare_group.add_argument('-vw', '--variant_window', required=False,
                        type=int, default=2, help='Number of variants in window to discard: default 2')
    compare_group.add_argument('--min_total_depth', type=str, dest="min_total_depth",
                              required=False, default=20, help='Minimum total depth to include a variant')
    compare_group.add_argument('--min_cov', type=str, dest="min_cov",
                              required=False, default=5, help='Minimum coverage required, indicated as uncovered "!"')
    compare_group.add_argument('--min_freq_include', type=str, dest="min_freq_include",
                              required=False, default=0.8, help='Minimum frequency to keep a variant')
    compare_group.add_argument('--min_freq_discard', type=str, dest="min_freq_discard",
                              required=False, default=0.2, help='Minimum frequency to include a low-depth variant')
    
    compare_group.add_argument('-C', '--complex', required=False, action='store_true', 
                               help='Disable complex/hotspot position filtering (default: filtering active)')
    compare_group.add_argument('--no_remove_faulty', required=False, action='store_true',
                                help='Disable faulty position and sample detection (default: enabled)')
    compare_group.add_argument('-R', '--remove_bed', required=False, type=str, default=False, 
                               help='BED file with positions to remove')
    compare_group.add_argument('-P', '--extract_bed', required=False, type=str, default=False, 
                               help='BED file with important positions or genes to annotate')
    compare_group.add_argument('-d', '--distance', default=15, required=False, type=int,
                               help='Minimun distance to cluster groups after comparison')
    compare_group.add_argument('-G', '--genomic', required=False, type=int, default=50,
                               help='SNP distance threshold for genomic network edges (default: 50)')
    compare_group.add_argument('--min_threshold_discard_uncov_sample', required=False,
                        type=float, default=0.8, help='min_threshold_discard_uncov_sample')
    compare_group.add_argument('--min_threshold_discard_uncov_pos', required=False,
                        type=float, default=0.8, help='min_threshold_discard_uncov_pos')
    compare_group.add_argument('--min_threshold_discard_htz_sample', required=False,
                        type=float, default=0.2, help='min_threshold_discard_htz_sample')
    compare_group.add_argument('--min_threshold_discard_htz_pos', required=False,
                        type=float, default=0.2, help='min_threshold_discard_htz_pos')
    compare_group.add_argument('--min_threshold_discard_all_pos', required=False,
                        type=float, default=1.2, help='min_threshold_discard_all_pos')
    compare_group.add_argument('--min_threshold_discard_all_sample', required=False,
                        type=float, default=1.2, help='min_threshold_discard_all_sample')

    arguments = parser.parse_args()

    return arguments
