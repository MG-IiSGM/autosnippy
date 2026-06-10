#!/usr/bin/env python

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import glob
import polars as pl
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import pairwise_distances, accuracy_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from pyvis.network import Network
import datetime
import time
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd  # pdist
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import Optional, Set, Tuple, List, Union, Dict


logger = logging.getLogger()


END_FORMATTING = '\033[0m'
WHITE_BG = '\033[0;30;47m'
NORMAL = '\033[22m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
BLUE = '\033[34m'
CYAN = '\033[36m'
YELLOW = '\033[93m'
DIM = '\033[2m'


def get_arguments():

    parser = argparse.ArgumentParser(
        prog='compare_snp_autosnippy.py', description='Pipeline to call variants (SNVs) with any non model organism. Specialised in Mycobacterium Tuberculosis')

    input_group = parser.add_argument_group('Input', 'input parameters')

    input_group.add_argument('-i', '--input', dest="input_dir", metavar="input_directory",
                        type=str, required=False, help='REQUIRED.Input directory containing all vcf files')
    input_group.add_argument('-s', '--sample_list', default=False, required=False,
                        help='File with sample names to analyse instead of all samples')
    input_group.add_argument('-o', '--output', type=str, required=True,
                        help='Name of all the output files, might include path')
    input_group.add_argument('-T', '--threads', type=int, dest="threads",
                              required=False, default=32, help='Threads to use')


    threshold_group = parser.add_argument_group('Threshold parameters', 'parameters for diferent threshold conditions')

    threshold_group.add_argument('-w', '--window', required=False,
                        type=int, default=10, help='Range of bases for variant accumulation to be considered as a hotspot: default 10')
    threshold_group.add_argument('-vw', '--variant_window', required=False,
                        type=int, default=2, help='Number of variants in window to discard: default 2')
    threshold_group.add_argument('--min_total_depth', type=str, dest="min_total_depth",
                              required=False, default=20, help='Minimum total depth to include a variant')
    threshold_group.add_argument('--min_cov', type=str, dest="min_cov",
                              required=False, default=5, help='Minimum coverage required, indicated as uncovered "!"')
    threshold_group.add_argument('--min_freq_include', type=str, dest="min_freq_include",
                              required=False, default=0.8, help='Minimum frequency to keep a variant')
    threshold_group.add_argument('--min_freq_discard', type=str, dest="min_freq_discard",
                              required=False, default=0.2, help='Minimum frequency to include a low-depth variant')


    compare_group = parser.add_argument_group('Compare', 'parameters for compare_snp')

    compare_group.add_argument('-c', '--only_compare', required=False,  action='store_true',
                               help='Skip comparison and run only ddtb_compare on existing matrix')
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


def check_create_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def import_variants_uncovPos_complex(
    sample: str,
    variant_file: str,
    cov_file: str,
    min_total_depth: int,
    min_cov: int,
    min_freq_include: float,
    min_freq_discard: float,
    window_size: int,
    max_variants_window: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:

    """
    Reads the variant file and coverage file for ONE sample using Polars.

    Returns
    -------
    dfv          : variants passing depth filters
    dfl          : low-depth but high-frequency variants  → '?'
    cov          : positions with insufficient coverage   → '!'
    complex_pos  : array of positions with complex variants (OLDVAR)
    hotspot_pos  : array of positions flagged as hotspot (>max_variants_window
                   variants within window_size bp, all passing min_total_depth
                   AND min_freq_discard simultaneously)
    """

    # ── Read schema ──────────────────────────────────────────────────────────
    schema = {
        'REGION'   : pl.String,
        'POS'      : pl.Int32,
        'REF'      : pl.String,
        'ALT'      : pl.String,
        'TOTAL_DP' : pl.Int16,
        'ALT_DP'   : pl.Int16,
        'ALT_FREQ' : pl.Float32,
        'TYPE'     : pl.String,
        'OLDVAR'   : pl.String,
    }

    df = pl.read_csv(
        variant_file,
        separator='\t',
        columns=list(schema.keys()),
        schema_overrides=schema,
        infer_schema_length=0,   # no inference, explicit schema used
        null_values=["", "NA", "NaN"],
    )

    if df.is_empty():
        raise ValueError(f"[{sample}] Empty or invalid variant file: {variant_file}")

    # Remove duplicates
    df = df.unique(subset=['POS', 'REF', 'ALT'], keep='first')

    # ── Base masks ───────────────────────────────────────────────────────────
    depth_ok = pl.col('TOTAL_DP') >= min_total_depth
    depth_low  = pl.col('TOTAL_DP') <  min_total_depth   # '<' to avoid overlap
    cov_ok     = pl.col('TOTAL_DP') >  min_cov          # above noise threshold
    freq_mask  = pl.col('ALT_FREQ') >= min_freq_include
    # snp_mask   = pl.col('TYPE')     == 'snp'

    # ── Complex positions ────────────────────────────────────────────────────
    complex_pos = (
        df
        .filter(pl.col('OLDVAR').is_not_null())
        .select('POS')
        .to_series()
        .cast(pl.Int32)
        .unique()
        .to_numpy()
    )

    # ── Main variants ────────────────────────────────────────────────────────
    dfv = (
        df
        .filter(depth_ok)
        .select(['REGION', 'POS', 'REF', 'ALT', 'ALT_FREQ'])
        .rename({'ALT_FREQ': sample})
        # Cast to String to unify type with later '?' and '!' markers
        .with_columns(pl.col(sample).cast(pl.String))
    )

    # ── Low-frequency variants ───────────────────────────────────────────────
    dfl = (
        df
        .filter(depth_low & cov_ok & freq_mask)
        .select(['REGION', 'POS', 'REF', 'ALT'])
        .with_columns(pl.lit('?').alias(sample))
    )

    # ── Uncovered positions ──────────────────────────────────────────────────
    cov_schema = {
        'REGION': pl.String,
        'POS'   : pl.Int32,
        sample  : pl.Int32,
    }

    if not os.path.exists(cov_file):
        logger.warning(RED + BOLD + f"[COV] " + NORMAL + f"Coverage file not found, skipping coverage filter: {sample}" + END_FORMATTING)
        cov = pl.DataFrame(schema={'REGION': pl.String, 'POS': pl.Int32, sample: pl.String})
    else:
        cov_raw = pl.read_csv(
            cov_file,
            separator='\t',
            has_header=False,
            new_columns=['REGION', 'POS', sample],
            schema_overrides=cov_schema,
        )
        cov = (
            cov_raw
            .filter(pl.col(sample) <= min_cov)
            .with_columns(pl.lit('!').alias(sample))
        )

    # ── Hotspot detection ────────────────────────────────────────────────────
    # A window is flagged as hotspot ONLY if ALL variants within it
    # simultaneously pass min_total_depth AND min_freq_discard.
    # If any variant in the window fails either threshold, the entire
    # window is disqualified — avoids removing real variants near noise.
    hotspot_pos: Set[int] = set()

    # Candidates: variants passing both depth and frequency thresholds
    candidates = df.filter(
        (pl.col('TOTAL_DP') >= min_total_depth) &
        (pl.col('ALT_FREQ') >= min_freq_discard)
    ).select(['POS', 'TOTAL_DP', 'ALT_FREQ'])

    # All variants in the file (used to check for disqualifying entries
    # within a window — any variant below either threshold disqualifies)
    all_variants = df.select(['POS', 'TOTAL_DP', 'ALT_FREQ'])

    if candidates.height > max_variants_window:
        cand_pos   = candidates['POS'].to_list()
        all_pos    = all_variants['POS'].to_list()
        all_dp     = all_variants['TOTAL_DP'].to_list()
        all_freq   = all_variants['ALT_FREQ'].to_list()

        for pos in cand_pos:

            # Collect all variants (not just candidates) within the window
            window_indices = [
                i for i, p in enumerate(all_pos)
                if abs(p - pos) <= window_size
            ]

            # Disqualify window if any variant fails either threshold
            any_below = any(
                all_dp[i] < min_total_depth or all_freq[i] < min_freq_discard
                for i in window_indices
            )

            if any_below:
                continue

            # Count candidates within the window
            window_candidates = [p for p in cand_pos if abs(p - pos) <= window_size]

            if len(window_candidates) > max_variants_window:
                hotspot_pos.update(window_candidates)
                logger.debug(
                    f"[HOTSPOT] {sample} | pos {pos} | "
                    f"{len(window_candidates)} variants in {window_size}bp "
                    f"window → flagged positions: {'; '.join(str(p) for p in sorted(window_candidates))}"
                )

    hotspot_array = np.array(sorted(hotspot_pos), dtype=np.int32)

    return dfv, dfl, cov, complex_pos, hotspot_array


def _process_sample(
    sample: str,
    variant_dir: str,
    coverage_dir: str,
    min_total_depth: int,
    min_cov: int,
    min_freq_include: float,
    min_freq_discard: float,
    window_size: int,
    max_variants_window: int,
) -> Optional[Tuple[str, pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]]:

    """
    Wrapper that calls import_variants_uncovPos_complex for one sample.
    Returns None if the sample is invalid (missing file or processing error).
    Designed to run inside a ThreadPoolExecutor.
    """

    variant_file = os.path.join(variant_dir, sample, "snps.all.ivar.tsv")
    cov_file     = os.path.join(coverage_dir, sample + ".cov")

    if not os.path.exists(variant_file):
        logger.warning(
            RED + BOLD + f"[SAMPLES] " + NORMAL + f"Variant file not found → skipping sample: {sample}" + END_FORMATTING + '\n'
        )
        return None

    try:
        dfv, dfl, cov, complex_pos, hotspot_pos = import_variants_uncovPos_complex(
            sample=sample,
            variant_file=variant_file,
            cov_file=cov_file,
            min_total_depth=min_total_depth,
            min_cov=min_cov,
            min_freq_include=min_freq_include,
            min_freq_discard=min_freq_discard,
            window_size=window_size,
            max_variants_window=max_variants_window,
        )
        logger.debug(f"[OK] {sample}")
        return sample, dfv, dfl, cov, complex_pos, hotspot_pos

    except Exception as exc:
        logger.warning(
            RED + f"[ERROR] {sample}: {exc}" + END_FORMATTING
        )
        return None


def ddbb_create_intermediate(
    variant_dir: str,
    coverage_dir: str,
    min_total_depth: int   = 12,
    min_cov: int           = 5,
    min_freq_discard: float = 0.2,
    min_freq_include: float = 0.7,
    apply_complex: bool     = False,
    window_size: int        = 10,
    max_variants_window: int = 2,
    samples                = False,
    max_workers: Optional[int] = None,
    complex_output_tsv: str = None,
) -> Tuple[pl.DataFrame, dict]:

    """
    Builds the variant comparison matrix across all samples.

    Pivot strategy
    --------------
    Instead of accumulating one column per sample (the 'diagonal' approach
    with group_by+agg, which scales poorly beyond ~100 columns), each sample
    reader returns data in LONG format with an explicit 'sample' column.
    A single native pl.DataFrame.pivot() call is made at the end, which is
    implemented in Rust and significantly more efficient.
 
    Parameters
    ----------
    variant_dir      : root directory containing per-sample subdirectories
    coverage_dir     : directory containing <sample>.cov files
    min_total_depth  : minimum total depth to include a variant
    min_freq_discard : minimum frequency to keep a variant (numeric cells only)
    min_freq_include : minimum frequency to include a low-depth variant as '?'
    min_cov          : minimum coverage; positions below this threshold → '!'
    apply_complex       : if True, zero out complex/hotspot positions per sample
                          before returning, preventing recovery in recalibration
    window_size         : bp half-window for hotspot detection (default 10)
    max_variants_window : variants in window to trigger hotspot flag (default 2)
    samples          : sample list, set, or path to a file; False → all samples
    max_workers      : threads for ThreadPoolExecutor (None = min(32, cpu_count))
    complex_output_tsv  : optional path for TSV (sample | position)

    Returns
    -------
    Tuple[pl.DataFrame, dict]
        pl.DataFrame : Position | N | Samples | <sample1> | <sample2> | ...
        dict         : {sample: set_of_int_positions} complex + hotspot per sample
    """

    # ── Sample selection ──────────────────────────────────────────────────────
    if samples and isinstance(samples, (set, list)) and len(samples) > 0:
        sample_iterable = set(samples)
    else:
        sample_iterable = {
            d for d in os.listdir(variant_dir)
            if os.path.isdir(os.path.join(variant_dir, d))
        }

    if not sample_iterable:
        logger.error(
            RED + BOLD + f"[FATAL] " + NORMAL + f"No sample directories found in: {variant_dir}" + END_FORMATTING
        )
        sys.exit(1)

    # ── Long-format accumulators ──────────────────────────────────────────────
    # Format: REGION | POS | REF | ALT | sample | value
    # A single pivot at the end replaces the per-column group_by+agg approach.
    variant_rows  : List[pl.DataFrame] = []
    lowfreq_rows  : List[pl.DataFrame] = []
    coverage_rows : List[pl.DataFrame] = []

    # Complex positions tracked per sample (not globally) so that a complex
    # position in sample A does not affect sample B downstream.
    # Format: {sample_name: set_of_int_positions}
    complex_positions_per_sample: dict = {}

    # ── Parallel sample loading ───────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:

        futures = {
            executor.submit(
                _process_sample,
                s,
                variant_dir,
                coverage_dir,
                min_total_depth,
                min_cov,
                min_freq_include,
                min_freq_discard,
                window_size,
                max_variants_window,
            ): s
            for s in sample_iterable
        }

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            sname, dfv, dfl, cov, complex_pos, hotspot_pos = result

            # Convert to long format by adding a 'sample' column
            # dfv already has a column named <sname>; rename it to 'value'
            if not dfv.is_empty():
                variant_rows.append(
                    dfv.rename({sname: 'value'})
                       .with_columns(pl.lit(sname).alias('sample'))
                )

            if not dfl.is_empty():
                lowfreq_rows.append(
                    dfl.rename({sname: 'value'})
                       .with_columns(pl.lit(sname).alias('sample'))
                )

            if not cov.is_empty():
                coverage_rows.append(
                    cov.rename({sname: 'value'})
                       .with_columns(pl.lit(sname).alias('sample'))
                )

            # Merge complex and hotspot into a single set per sample
            combined = set(complex_pos.tolist()) | set(hotspot_pos.tolist())
            if combined:
                complex_positions_per_sample[sname] = combined

    # ── Final check ───────────────────────────────────────────────────────────
    if not variant_rows:
        logger.error(RED + BOLD + "[FATAL] " + NORMAL + f"No valid samples found after filtering." + END_FORMATTING)
        return pl.DataFrame()

    index_cols = ['REGION', 'POS', 'REF', 'ALT']

    # ── Long concat + native pivot ────────────────────────────────────────────
    dfv_long = pl.concat(variant_rows, how='vertical')

    # Apply min_freq_discard only to numeric cells (preserves '?' and '!')
    dfv_long = dfv_long.with_columns(
        pl.when(pl.col('value').is_in(['?', '!']))
        .then(pl.col('value'))
        .when(pl.col('value').cast(pl.Float32, strict=False) >= min_freq_discard)
        .then(pl.col('value'))
        .otherwise(pl.lit(None))
        .alias('value')
    ).filter(pl.col('value').is_not_null())

    # Native Polars pivot: rows=position, columns=sample, values=value
    df = dfv_long.pivot(
        values='value',
        index=index_cols,
        on='sample',
        aggregate_function='first',
    )

    all_sample_cols = [c for c in df.columns if c not in index_cols]

    # ── Low-freq patch ────────────────────────────────────────────────────────
    # '?' fills nulls at positions where the variant failed the depth filter
    # but passed the frequency threshold.
    if lowfreq_rows:
        dfl_long = pl.concat(lowfreq_rows, how='vertical')

        dfl_pivot = dfl_long.pivot(
            values='value',
            index=index_cols,
            on='sample',
            aggregate_function='first',
        )

        # Join + coalesce: '?' fills nulls where no main variant was present
        df = df.join(dfl_pivot, on=index_cols, how='left', suffix='_lf')

        for s in all_sample_cols:
            lf_col = s + '_lf'
            if lf_col in df.columns:
                df = df.with_columns(
                    pl.coalesce([pl.col(s), pl.col(lf_col)]).alias(s)
                ).drop(lf_col)

    # ── INDELs: compute positions to exclude from coverage patch ─────────────
    indel_mask_expr = (
        (pl.col('REF').str.len_chars() > 1) |
        (pl.col('ALT').str.len_chars() > 1)
    )

    indel_rows_df = df.filter(indel_mask_expr).select(['POS', 'REF'])

    if indel_rows_df.height > 0:
        pos_arr = indel_rows_df['POS'].to_numpy()
        len_arr = indel_rows_df['REF'].str.len_chars().to_numpy()

        indel_positions_final: Set[int] = set()
        for pos, l in zip(pos_arr, len_arr):
            for off in range(-(5 + int(l)), (5 + int(l))):
                indel_positions_final.add(int(pos) + off)
    else:
        indel_positions_final = set()

    # ── Coverage patch (chunked to avoid OOM with >2000 samples) ─────────────
    # '!' fills nulls at positions with insufficient coverage, excluding positions near INDELs.
    # Instead of concatenating all coverage rows into a single giant DataFrame
    # before pivoting, we process coverage in batches of 200 samples at a time,
    # applying each batch's join+coalesce incrementally onto df.
    # This keeps peak memory proportional to batch_size, not total samples.
    if coverage_rows:
        cov_batch_size = 200  # tune this down if still hitting OOM

        for batch_start in range(0, len(coverage_rows), cov_batch_size):
            batch = coverage_rows[batch_start: batch_start + cov_batch_size]

            cov_long = pl.concat(batch, how='vertical')

            if indel_positions_final:
                cov_long = cov_long.filter(
                    ~pl.col('POS').is_in(list(indel_positions_final))
                )

            cov_pivot = cov_long.pivot(
                values='value',
                index=['REGION', 'POS'],
                on='sample',
                aggregate_function='first',
            )

            # Identify which sample columns this batch covers
            batch_sample_cols = [
                c for c in cov_pivot.columns
                if c not in ['REGION', 'POS']
            ]

            df = df.join(cov_pivot, on=['REGION', 'POS'], how='left', suffix='_cov')

            for s in batch_sample_cols:
                cov_col = s + '_cov'
                if cov_col in df.columns:
                    df = df.with_columns(
                        pl.coalesce([pl.col(s), pl.col(cov_col)]).alias(s)
                    ).drop(cov_col)

            # Explicitly free batch memory before next iteration
            del cov_long, cov_pivot, batch

    # ── Fill remaining nulls → '0' ────────────────────────────────────────────
    df = df.with_columns([
        pl.col(s).fill_null('0') for s in all_sample_cols
    ])

    # ── Zero out complex/hotspot positions per sample ─────────────────────────
    # Performed before N/Samples computation and before recalibration so
    # that these positions cannot be recovered downstream.
    # Only active when apply_complex=True (controlled by args.complex).
    if apply_complex and complex_positions_per_sample:

        complex_rows: List[Tuple[str, str]] = []

        # Parse POS once for all rows
        positions_int = (
            df.select(pl.col('POS').cast(pl.Int32))['POS'].to_list()
        )
        position_keys = (
            (df['REGION'] + '|' + df['REF'] + '|' +
             df['POS'].cast(pl.String) + '|' + df['ALT'])
            .to_list()
        )

        for s in all_sample_cols:
            blocked = complex_positions_per_sample.get(s, set())
            if not blocked:
                continue

            col_vals = df[s].to_list()
            updated  = False

            for i, (val, pos) in enumerate(zip(col_vals, positions_int)):
                if pos in blocked and str(val) not in ('0', '!'):
                    col_vals[i] = '0'
                    complex_rows.append((s, position_keys[i]))
                    updated = True
                    logger.debug(
                        f"[COMPLEX] {s} | pos {pos} | {val} → '0' (complex/hotspot)"
                    )

            if updated:
                df = df.with_columns(pl.Series(s, col_vals, dtype=pl.String))
                logger.debug(f"[COMPLEX] {s} | positions zeroed: {sum(1 for a in complex_rows if a[0] == s)}")

        # Write TSV if requested
        if complex_output_tsv and complex_rows:
            # Group positions by sample: one row per sample, positions
            # extracted from Position string (field index 2) and
            # separated by '; '
            from collections import defaultdict
            sample_positions: dict = defaultdict(list)
            for sample_name, pos_str in complex_rows:
                pos_num = pos_str.split('|')[2]
                sample_positions[sample_name].append(pos_num)

            pl.DataFrame({
                'sample'  : list(sample_positions.keys()),
                'n_positions': [
                    len(positions)
                    for positions in sample_positions.values()
                ],
                'position': [
                    '; '.join(positions)
                    for positions in sample_positions.values()
                ],
            }).write_csv(complex_output_tsv, separator='\t')
            logger.info(
                '\n' + GREEN + BOLD + f"[COMPLEX] " + NORMAL + f"Complex_hotspot tsv written: {complex_output_tsv} "
                f"({len(complex_rows)} entries)" + END_FORMATTING + '\n'
            )

    # ── N and Samples columns ─────────────────────────────────────────────────
    # A cell is 'present' if its value is not '!', '0', or null
    present_flags = [
        (~pl.col(s).is_in(['!', '0'])).alias(f'__p_{s}')
        for s in all_sample_cols
    ]
    df = df.with_columns(present_flags)

    flag_cols = [f'__p_{s}' for s in all_sample_cols]

    # N = number of samples where the variant is present
    df = df.with_columns(
        pl.sum_horizontal(flag_cols).cast(pl.Int32).alias('N')
    )

    # Samples = comma-separated list of sample names where variant is present
    sample_names   = np.array(all_sample_cols)
    present_matrix = df.select(flag_cols).to_numpy().astype(bool)
    samples_col    = [",".join(sample_names[row]) for row in present_matrix]

    df = (
        df.with_columns(pl.Series('Samples', samples_col))
          .drop(flag_cols)
          .filter(pl.col('N') > 0)
    )

    # ── Position column and final reorder ─────────────────────────────────────
    df = df.with_columns(
        (
            pl.col('REGION') + '|' +
            pl.col('REF')    + '|' +
            pl.col('POS').cast(pl.String) + '|' +
            pl.col('ALT')
        ).alias('Position')
    ).drop(index_cols)

    remaining = [c for c in df.columns if c not in ['Position', 'N', 'Samples']]
    df = df.select(['Position', 'N', 'Samples'] + remaining)

    return df, complex_positions_per_sample


def remove_position_range(
    df: pl.DataFrame,
) -> pl.DataFrame:

    """
    Removes SNP positions that fall within the genomic range covered by any
    INDEL present in the matrix (REF or ALT with 2+ characters).

    For each INDEL, the excluded range is [POS+1, POS+len(REF)-1], i.e. the
    positions immediately downstream of the INDEL start that are covered by
    the deleted/inserted sequence.

    Only pure SNP positions are removed — INDEL rows themselves are kept.

    Parameters
    ----------
    df : pl.DataFrame from recalibrate_ddbb_vcf_intermediate,
         OR path string to the recalibrated TSV file

    Returns
    -------
    pl.DataFrame with SNPs inside INDEL ranges removed
    """

    # ── Parse position components ─────────────────────────────────────────────
    # Position format: REGION|REF|POS|ALT
    df = df.with_columns([
        pl.col('Position').str.split('|').list.get(1).alias('_REF'),
        pl.col('Position').str.split('|').list.get(2).cast(pl.Int32).alias('_POS'),
        pl.col('Position').str.split('|').list.get(3).alias('_ALT'),
    ])

    # ── Identify INDEL rows: REF or ALT with 2+ characters ───────────────────
    indel_mask = (
        (pl.col('_REF').str.len_chars() >= 2) |
        (pl.col('_ALT').str.len_chars() >= 2)
    )

    indels = df.filter(indel_mask).select(['_POS', '_REF'])

    if indels.is_empty():
        logger.info(RED + BOLD + "[INDEL] " + NORMAL + "No INDELs found in matrix — nothing to remove." + END_FORMATTING)
        return df.drop(['_REF', '_POS', '_ALT'])

    # ── Build excluded position set vectorially ───────────────────────────────
    # For each INDEL at POS with REF of length L, exclude positions
    # [POS+1, POS+L-1] — the range covered by the deleted sequence.
    # This is computed in numpy for speed and converted to a Polars filter.
    pos_arr = indels['_POS'].to_numpy()
    len_arr = indels['_REF'].str.len_chars().to_numpy()

    excluded: Set[int] = set()
    for pos, l in zip(pos_arr, len_arr):
        for p in range(int(pos) + 1, int(pos) + int(l)):
            excluded.add(p)

    if not excluded:
        return df.drop(['_REF', '_POS', '_ALT'])

    logger.debug(f"[INDEL] {len(indels)} INDEL(s) found → excluding {len(excluded)} downstream positions")

    # ── Remove SNP positions that fall within any INDEL range ─────────────────
    # A position is removed only if:
    #   1. Its POS falls within an INDEL-covered range, AND
    #   2. It is itself a pure SNP (not an INDEL row) — INDEL rows are kept.
    snp_in_indel = (~indel_mask) & (pl.col('_POS').is_in(list(excluded)))

    removed = df.filter(snp_in_indel)
    kept    = df.filter(~snp_in_indel)

    if not removed.is_empty():
        for pos in removed['Position'].to_list():
            logger.debug(f"[INDEL] Removed SNP inside INDEL range: {pos}")
        logger.info(
            GREEN + BOLD + f"[INDEL] " + NORMAL + f"{removed.height} SNP position(s) removed as they fall within INDEL ranges" + END_FORMATTING + '\n'
        )

    return kept.drop(['_REF', '_POS', '_ALT'])


def revised_df(
    df: Union[pl.DataFrame, str],
    out_dir: str                                = None,
    symbol_lowcov: pl.DataFrame                = None,
    min_freq_include: float                    = 0.8,
    min_freq_discard: float                    = 0.2,
    min_threshold_discard_uncov_sample: float  = 0.5,
    min_threshold_discard_uncov_pos: float     = 0.5,
    min_threshold_discard_htz_sample: float    = 0.5,
    min_threshold_discard_htz_pos: float       = 0.5,
    min_threshold_discard_all_pos: float       = 0.5,
    min_threshold_discard_all_sample: float    = 0.5,
    remove_faulty: bool                        = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    """
    Final cleanup and binarisation of the recalibrated SNP matrix.

    Faulty position detection (row-wise)
    ------------------------------------
    Uncovered score:
        numeric / '0' → 1.0  (well covered, wildtype or variant)
        '?'           → 0.5  (low coverage, uncertain)
        '!'           → 0.0  (no coverage)
      Row mean < min_threshold_discard_uncov_pos → discard position

    Heterozygous score (excludes '!' and '?' from calculation):
        > min_freq_include or < min_freq_discard → 1.0  (fixed)
        between min_freq_discard and min_freq_include → 0.0 (heterozygous)
      Row nanmean < min_threshold_discard_htz_pos → discard position

    Faulty position: discarded if uncov OR htz threshold exceeded,
      OR if uncov + htz sum < min_threshold_discard_all_pos

    Faulty sample detection (column-wise, same logic)
    --------------------------------------------------
    Uncovered score per sample:
        numeric / '0' → 1.0 | '?' and '!' → 0.0
    Heterozygous score per sample:
        fixed → 1.0 | heterozygous → 0.0 | '!' and '?' excluded (NaN)
    Faulty sample: discarded if uncov OR htz threshold exceeded,
      OR if uncov + htz sum < min_threshold_discard_all_sample

    Final binarisation
    ------------------
      '?' or value > min_freq_include                      → '1'
      value between min_freq_discard and min_freq_include  → '0.5'
      '!' or value <= min_freq_discard                     → '0'

    N and Samples are recomputed after binarisation.
    '0.5' counts as present and is included in N and Samples.

    Low-coverage filter
    -------------------
    Positions present in symbol_lowcov are removed from the full matrix
    before splitting into SNPs, so the filter is applied only once.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        full_df : complete final matrix
        snps_df : SNPs only (REF and ALT both single characters)
    """

    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in df.columns if c not in index_cols]

    logger.info(
        GREEN + BOLD + f"[REVISED] " + NORMAL + f"Matrix loaded: {df.height} positions × "
        f"{len(sample_cols)} samples" + END_FORMATTING +'\n'
    )

    # ═════════════════════════════════════════════════════════════════════════
    # FAULTY DETECTION
    # ═════════════════════════════════════════════════════════════════════════

    if remove_faulty:

        # ── Extract raw values as numpy for vectorised score computation ──────
        # Mixed-type string columns are handled more efficiently in numpy
        # than in Polars for row/column-wise operations with conditionals.
        values = df.select(sample_cols).to_numpy().astype(str)

        n_pos     = values.shape[0]
        n_samples = values.shape[1]

        with np.errstate(all='ignore'):

            # ── Uncovered score matrix ────────────────────────────────────────────
            # Encodes coverage quality per cell:
            #   '!' → 0.0 (no coverage)
            #   '?' → 0.5 (low coverage, uncertain)
            #   anything else → 1.0 (well covered, wildtype or variant)
            uncov_matrix = np.where(
                values == '!', 0.0,
                np.where(values == '?', 0.5, 1.0)
            ).astype(float)

            # ── Heterozygous score matrix ─────────────────────────────────────────
            # Encodes fixation quality per cell, excluding '!' and '?' (→ NaN):
            #   > min_freq_include or < min_freq_discard → 1.0 (fixed)
            #   between thresholds                       → 0.0 (heterozygous)
            #   '!' or '?'                               → NaN (excluded)
            # nanmean is used downstream to ignore NaN cells.
            htz_matrix = np.full((n_pos, n_samples), np.nan)

            for i in range(n_pos):
                for j in range(n_samples):
                    v = values[i, j]
                    if v in ('!', '?', '0', 'None'):
                        continue
                    try:
                        f = float(v)
                        if f > min_freq_include or f < min_freq_discard:
                            htz_matrix[i, j] = 1.0
                        else:
                            htz_matrix[i, j] = 0.0
                    except ValueError:
                        continue

            # ── Per-position scores ───────────────────────────────────────────────
            # uncov_pos: mean coverage quality across all samples for each position
            # htz_pos  : mean fixation quality across numeric-only cells per position
            #            rows where all cells are NaN (all '!'/'?') → set to 1.0
            #            to avoid discarding positions with no numeric evidence
            uncov_pos = uncov_matrix.mean(axis=1)
            htz_pos = np.nanmean(htz_matrix, axis=1)
            htz_pos = np.nan_to_num(htz_pos, nan=1.0)

            report_position = pl.DataFrame({
                'Position'   : df['Position'].to_list(),
                'uncov_fract': uncov_pos.tolist(),
                'htz_frac'   : htz_pos.tolist(),
                'faulty_frac': (uncov_pos + htz_pos).tolist(),
            })

            # A position is faulty if ANY of the three conditions is met:
            #   - uncov mean below threshold (too many uncovered/uncertain samples)
            #   - htz mean below threshold (too many heterozygous samples)
            #   - combined score below threshold
            faulty_positions = report_position.filter(
                (pl.col('uncov_fract') < min_threshold_discard_uncov_pos) |
                (pl.col('htz_frac')    < min_threshold_discard_htz_pos)   |
                (pl.col('faulty_frac') < min_threshold_discard_all_pos)
            )['Position'].to_list()

            # ── Per-sample scores ─────────────────────────────────────────────────
            # uncov_sample: mean coverage quality per sample across all positions
            #   numeric/'0' → 1.0 | '?' → 0.0 | '!' → 0.0
            #   ('?' treated as 0 here: sample-level uncov is stricter than pos-level)
            uncov_sample = np.where(
                (values == '!') | (values == '?'), 0.0, 1.0
            ).astype(float).mean(axis=0)

            # htz_sample: mean fixation quality per sample across numeric-only cells
            htz_sample = np.nanmean(htz_matrix, axis=0)
            htz_sample = np.nan_to_num(htz_sample, nan=1.0)

            report_samples = pl.DataFrame({
                'sample'     : sample_cols,
                'uncov_fract': uncov_sample.tolist(),
                'htz_frac'   : htz_sample.tolist(),
                'faulty_frac': (uncov_sample + htz_sample).tolist(),
            })

            # A sample is faulty if ANY of the three conditions is met
            faulty_samples = report_samples.filter(
                (pl.col('uncov_fract') < min_threshold_discard_uncov_sample) |
                (pl.col('htz_frac')    < min_threshold_discard_htz_sample)   |
                (pl.col('faulty_frac') < min_threshold_discard_all_sample)
            )['sample'].to_list()

            logger.info(
                GREEN + BOLD + f"[REVISED] " + NORMAL + f"Faulty positions: {len(faulty_positions)} | "
                f"Faulty samples: {len(faulty_samples)}" + END_FORMATTING +'\n'
            )

        # ── Export QC reports ─────────────────────────────────────────────────
        if out_dir:
            out_dir = os.path.abspath(out_dir)

            report_position.write_csv(
                os.path.join(out_dir, 'report_positions.tsv'), separator='\t'
            )
            report_samples.write_csv(
                os.path.join(out_dir, 'report_samples.tsv'), separator='\t'
            )
            pl.DataFrame({'position': faulty_positions}).write_csv(
                os.path.join(out_dir, 'faulty_positions.tsv'), separator='\t'
            )
            pl.DataFrame({'sample': faulty_samples}).write_csv(
                os.path.join(out_dir, 'faulty_samples.tsv'), separator='\t'
            )

            logger.info(GREEN + BOLD + f"[REVISED] " + NORMAL + f"Reports written to: {out_dir}" + END_FORMATTING + '\n')

        # ── Apply faulty filters ──────────────────────────────────────────────
        if remove_faulty and faulty_positions:
            df = df.filter(~pl.col('Position').is_in(faulty_positions))
            sample_cols = [c for c in df.columns if c not in index_cols]
            logger.info(
                GREEN + BOLD + f"[REVISED] " + NORMAL + f"{len(faulty_positions)} faulty position(s) removed" + END_FORMATTING
            )

        if remove_faulty and faulty_samples:
            df = df.drop([s for s in faulty_samples if s in df.columns])
            sample_cols = [c for c in df.columns if c not in index_cols]
            logger.info(
                GREEN + BOLD + f"[REVISED] " + NORMAL + f"{len(faulty_samples)} faulty sample(s) removed" + END_FORMATTING +'\n'
            )

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL BINARISATION
    # ═════════════════════════════════════════════════════════════════════════

    # Convert all cell values to a three-state encoding:
    #   '1'   → variant present and well-supported (or '?': low-cov but detected)
    #   '0.5' → heterozygous / intermediate frequency
    #   '0'   → absent, noise, or no coverage
    binarise_exprs = []
    for s in sample_cols:
        binarise_exprs.append(
            pl.when(pl.col(s) == '?')
            .then(pl.lit('1'))
            .when(pl.col(s) == '!')
            .then(pl.lit('0'))
            .when(pl.col(s).cast(pl.Float32, strict=False) > min_freq_include)
            .then(pl.lit('1'))
            .when(pl.col(s).cast(pl.Float32, strict=False) >= min_freq_discard)
            .then(pl.lit('0.5'))
            .otherwise(pl.lit('0'))
            .alias(s)
        )

    df = df.with_columns(binarise_exprs)

    # ── Export intermediate cleaned file ──────────────────────────────────────
    if out_dir:
        df.write_csv(
            os.path.join(out_dir, 'intermediate.highfreq.tsv'), separator='\t'
        )

    # ── Recompute N and Samples after binarisation ────────────────────────────
    # '0.5' is treated as present — included in both N and Samples.
    # Only '0' and '!' are considered absent.
    present_flags = [
        (~pl.col(s).is_in(['0', '!'])).alias(f'__p_{s}')
        for s in sample_cols
    ]
    df = df.with_columns(present_flags)
    flag_cols = [f'__p_{s}' for s in sample_cols]

    df = df.with_columns(
        pl.sum_horizontal(flag_cols).cast(pl.Int32).alias('N')
    )

    sample_names   = np.array(sample_cols)
    present_matrix = df.select(flag_cols).to_numpy().astype(bool)
    samples_col    = [",".join(sample_names[row]) for row in present_matrix]

    df = (
        df.with_columns(pl.Series('Samples', samples_col))
          .drop(flag_cols)
          .filter(pl.col('N') > 0)
    )

    df = df.select(index_cols + sample_cols)

    logger.info(
        GREEN + BOLD + f"[REVISED] " + NORMAL + f"Final matrix: {df.height} positions × "
        f"{len(sample_cols)} samples" + END_FORMATTING + '\n'
    )

    # ═════════════════════════════════════════════════════════════════════════
    # LOW-COVERAGE FILTER
    # ═════════════════════════════════════════════════════════════════════════

    # Remove positions identified as having uniform calls but incomplete
    # coverage (from extract_lowcov). Applied once on the full df before
    # splitting so the filter is not repeated on the SNPs subset.
    if symbol_lowcov is not None and not symbol_lowcov.is_empty():
        lowcov_positions = symbol_lowcov['Position'].to_list()
        before = df.height
        df = df.filter(~pl.col('Position').is_in(lowcov_positions))
        logger.info(
            GREEN + BOLD + f"[REVISED] " + NORMAL + f"{before - df.height} low-coverage position(s) removed" + END_FORMATTING + '\n'
        )

    # ═════════════════════════════════════════════════════════════════════════
    # SPLIT: FULL vs SNPs ONLY
    # ═════════════════════════════════════════════════════════════════════════

    # SNPs: both REF (Position field index 1) and ALT (field index 3)
    # must be exactly one character — excludes all INDELs.
    snps_df = df.filter(
        (pl.col('Position').str.split('|').list.get(1).str.len_chars() == 1) &
        (pl.col('Position').str.split('|').list.get(3).str.len_chars() == 1)
    )

    logger.info(
        GREEN + BOLD + f"[REVISED] " + NORMAL + f"SNPs only: {snps_df.height} positions "
        f"({df.height - snps_df.height} INDELs excluded)" + END_FORMATTING + '\n'
    )

    return df, snps_df


def extract_lowcov(
    df: pl.DataFrame,
    min_freq_include: float = 0.8,
) -> pl.DataFrame:

    """
    Extracts positions where all samples with sufficient coverage agree
    (all present or all absent), but at least one sample has '!' or '?'.

    These are positions where the variant call is consistent among covered
    samples, but the result is incomplete due to low or missing coverage
    in some samples — useful for downstream QC and reporting.

    Logic:
      1. Numeric values > min_freq_include → 1 (variant present)
         Numeric values <= min_freq_include → 0 (variant absent)
      2. '!' and '?' are excluded from the agreement check
      3. A row is kept if:
         - All numeric cells agree (all 0 or all 1), AND
         - At least one cell is '!' or '?'

    Parameters
    ----------
    df              : pl.DataFrame or path to TSV
    min_freq_include: frequency threshold above which a variant is
                      considered present (default 0.5)

    Returns
    -------
    pl.DataFrame containing only the rows that meet the criteria above
    """

    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in df.columns if c not in index_cols]

    # ── Binarise numeric cells ────────────────────────────────────────────────
    # For each sample column, create a binarised version:
    #   - numeric > min_freq_include → '1'
    #   - numeric <= min_freq_include → '0'
    #   - '!' or '?' → kept as-is (excluded from agreement check)
    # This mirrors the legacy fn() lambda but vectorised across all columns.
    bin_exprs = []
    for s in sample_cols:
        bin_exprs.append(
            pl.when(pl.col(s).is_in(['!', '?']))
            .then(pl.col(s))
            .when(pl.col(s).cast(pl.Float32, strict=False) > min_freq_include)
            .then(pl.lit('1'))
            .otherwise(pl.lit('0'))
            .alias(f'_bin_{s}')
        )

    df_bin = df.with_columns(bin_exprs)
    bin_cols = [f'_bin_{s}' for s in sample_cols]

    # ── Check agreement among numeric cells ───────────────────────────────────
    # A row 'agrees' if all numeric cells (those not '!' or '?') are equal.
    # We do this by collecting unique numeric values per row and checking
    # that there is exactly one unique value (all 0 or all 1).
    # Computed row-by-row in Python since Polars lacks a native row-wise
    # n_unique for mixed-type string columns.
    bin_matrix   = df_bin.select(bin_cols).to_numpy()
    has_symbol   = []
    all_agree    = []

    for row in bin_matrix:
        numeric_vals = [v for v in row if v not in ('!', '?')]
        symbol_vals  = [v for v in row if v in ('!', '?')]

        has_symbol.append(len(symbol_vals) > 0)
        # Agreement: all numeric cells have the same value (or none exist)
        all_agree.append(len(set(numeric_vals)) <= 1)

    df_bin = df_bin.with_columns([
        pl.Series('_has_symbol', has_symbol),
        pl.Series('_all_agree',  all_agree),
    ])

    # ── Filter: agree AND has at least one '!' or '?' ─────────────────────────
    result = (
        df_bin
        .filter(pl.col('_all_agree') & pl.col('_has_symbol'))
        .select(df.columns)   # drop all auxiliary columns
    )

    # ── Binarise sample columns in the output ─────────────────────────────────
    # Numeric values > min_freq_include → '1', <= min_freq_include → '0'.
    # '!' and '?' are preserved as-is for identification purposes.
    binarise_output = [
        pl.when(pl.col(s).is_in(['!', '?']))
        .then(pl.col(s))
        .when(pl.col(s).cast(pl.Float32, strict=False) > min_freq_include)
        .then(pl.lit('1'))
        .otherwise(pl.lit('0'))
        .alias(s)
        for s in sample_cols
    ]
    result = result.with_columns(binarise_output)

    logger.info(
        GREEN + BOLD + f"[LOWCOV] " + NORMAL + f"{result.height} position(s) identified with "
        f"uniform calls but incomplete coverage" + END_FORMATTING + '\n'
    )

    return result


def recheck_variant_rawvcf_intermediate(
    sample: str,
    variant_folder: str,
    positions: List[str],
    alt_snps: List[str],
    sample_values: List[str],
    min_total_depth: int,
    min_cov: int,
    min_freq_include: float,
    min_freq_discard: float,
    complex_window: int = 5,
    blocked_positions: Set[int] = None,
) -> Tuple[str, dict]:

    """
    Recalibrates the variant calls for ONE sample by reading its raw
    FreeBayes VCF and attempting to recover positions currently marked
    as '0' in the intermediate matrix.

    Recovery logic (mirrors import_variants_uncovPos_complex thresholds):
      - TOTAL_DP >= min_total_depth + allele match + freq >= min_freq_discard
                                                        → numeric frequency
      - TOTAL_DP >= min_total_depth + allele match + freq < min_freq_discard
                                                        → '0' (noise)
      - min_cov < TOTAL_DP < min_total_depth
        + ALT_FREQ >= min_freq_include                  → '?'
      - TOTAL_DP > 0 but no threshold met               → '!'
      - Position absent from VCF                        → stays '0'

    Positions in blocked_positions are permanently excluded from recovery
    regardless of VCF evidence — these are complex/hotspot positions
    flagged during ddbb_create_intermediate.

    Complex variants (FreeBayes 'complex' INFO tag) are handled within a
    ±complex_window bp window around the VCF position.

    Parameters
    ----------
    sample            : sample name
    variant_folder    : root directory containing per-sample subdirectories
    positions         : list of POS strings aligned with matrix rows
    alt_snps          : list of ALT strings aligned with matrix rows
    sample_values     : current row values for this sample aligned with positions
    min_total_depth   : minimum depth for a confident variant call
    min_cov           : minimum depth floor; below this → '!'
    min_freq_include  : minimum ALT_FREQ threshold for '?' assignment
    min_freq_discard  : minimum ALT_FREQ to recover a numeric value
    complex_window    : bp half-window around complex variants (default ±5)
    blocked_positions : set of int positions permanently blocked from recovery
                        (complex/hotspot); None → no blocking

    Returns
    -------
    Tuple of (sample_name, {row_index: new_value})
        Only positions that changed are included in the updates dict.
    """

    vcf_file = os.path.join(variant_folder, sample, "snps.raw.vcf")

    if not os.path.exists(vcf_file):
        logger.warning(
            RED + BOLD + f"[RECAL] " + NORMAL + f"Raw VCF not found, skipping recalibration: {sample}" + END_FORMATTING
        )
        return sample, {}

    # ── Build lookup of positions to recalibrate ──────────────────────────────
    # Only cells currently equal to '0' are candidates for recovery.
    # Using a dict {pos_string: row_index} allows O(1) lookup while
    # parsing the VCF line by line, avoiding any nested loops.
    zero_positions: dict = {
        positions[i]: i
        for i, v in enumerate(sample_values)
        if str(v) == '0'
    }

    # ── Remove blocked positions from recovery candidates ─────────────────────
    # Positions flagged as complex/hotspot during ddbb_create_intermediate
    # are permanently excluded from recovery regardless of VCF evidence.
    # By removing them from zero_positions here, the VCF parser below
    # will simply skip them even if it finds supporting evidence.
    blocked_zero_positions: dict = {}
    if blocked_positions:
        blocked_zero_positions = {
            positions[i]: i
            for i, v in enumerate(sample_values)
            if str(v) == '0' and int(positions[i]) in blocked_positions
        }

        zero_positions = {
            pos: idx for pos, idx in zero_positions.items()
            if int(pos) not in blocked_positions
        }

        logger.debug(
            f"[RECAL] {sample} | {len(blocked_positions)} complex/hotspot "
            f"positions: {len(blocked_zero_positions)} are '0' candidates "
            f"for conservative recovery"
        )

    if not zero_positions and not blocked_zero_positions:
        # Nothing to recover for this sample
        return sample, {}

    updates: dict = {}
    recovered_numeric = 0
    recovered_lowfreq = 0
    recovered_lowcov  = 0
    recovered_complex  = 0

    try:
        with open(vcf_file, 'r') as f:
            for line in f:

                # Skip VCF header lines
                if line.startswith('#'):
                    continue

                fields = line.split('\t')
                if len(fields) < 10:
                    continue

                vcf_pos = fields[1]
                vcf_alt = fields[4]
                fmt     = fields[8].split(':')
                val     = fields[9].strip().split(':')

                # ── Parse FreeBayes-specific depth fields ─────────────────
                # FreeBayes uses DP (total depth) and AO (alt observation
                # count) in the FORMAT column. AO can be comma-separated
                # for multi-allelic sites — in that case we take the maximum.
                try:
                    dp_idx    = fmt.index('DP')
                    ao_idx    = fmt.index('AO')
                    vcf_depth = int(val[dp_idx])
                    ao_raw = val[ao_idx].split(',')
                    if len(ao_raw) > 1:
                        # Multi-allelic: take the most supported alt allele
                        vcf_alt_depth = max(int(x) for x in ao_raw)
                        vcf_alt       = vcf_alt.split(',')[-1]
                    else:
                        vcf_alt_depth = int(ao_raw[0])

                    vcf_alt_freq = (
                        round(vcf_alt_depth / vcf_depth, 4)
                        if vcf_depth > 0 else 0.0
                    )

                except (ValueError, IndexError):
                    logger.debug(
                        f"[RECAL] Could not parse FORMAT fields at pos "
                        f"{vcf_pos} in {sample} — skipping line"
                    )
                    continue

                # ── Standard position recovery ────────────────────────────
                # Check if this VCF position matches any '0' cell we need
                # to recalibrate. Allele matching covers three cases:
                #   1. Exact SNP match (alt_snp == vcf_alt)
                #   2. INDEL: alt_snp longer than 1 bp with sufficient freq
                #   3. Complex: alt_snp is contained within vcf_alt string
                if vcf_pos in zero_positions:
                    idx     = zero_positions[vcf_pos]
                    alt_snp = alt_snps[idx]

                    allele_match = (
                        alt_snp == vcf_alt or
                        (len(alt_snp) > 1 and vcf_alt_freq >= min_freq_include) or
                        alt_snp in vcf_alt
                    )

                    if allele_match:
                        if vcf_depth >= min_total_depth:
                            if vcf_alt_freq >= min_freq_discard:
                                # Well-supported variant: recover numeric frequency
                                updates[idx] = str(vcf_alt_freq)
                                recovered_numeric += 1
                                logger.debug(
                                    f"[RECAL] {sample} | pos {vcf_pos} | "
                                    f"0 → {vcf_alt_freq} (DP={vcf_depth}, "
                                    f"FREQ={vcf_alt_freq})"
                                )
                            else:
                                # Depth OK but frequency below noise threshold
                                updates[idx] = '0'
                                logger.debug(
                                    f"[RECAL] {sample} | pos {vcf_pos} | "
                                    f"depth OK but freq {vcf_alt_freq} "
                                    f"< min_freq_discard → '0'"
                                )

                        elif vcf_depth > min_cov and vcf_alt_freq >= min_freq_include:
                            # Intermediate depth with good frequency: low-confidence
                            updates[idx] = '?'
                            recovered_lowfreq += 1
                            logger.debug(
                                f"[RECAL] {sample} | pos {vcf_pos} | "
                                f"0 → '?' (DP={vcf_depth}, FREQ={vcf_alt_freq})"
                            )

                        else:
                            # Present but below all thresholds: low coverage
                            updates[idx] = '!'
                            recovered_lowcov += 1
                            logger.debug(
                                f"[RECAL] {sample} | pos {vcf_pos} | "
                                f"0 → '!' (DP={vcf_depth}, FREQ={vcf_alt_freq})"
                            )

                # ── Conservative recovery of complex positions ────────────
                # Same depth requirements as the normal flow, but 
                # the minimum frequency is min_freq_include (stricter than 
                # min_freq_discard) to filter out cross-platform noise.
                if vcf_pos in blocked_zero_positions:
                    idx     = blocked_zero_positions[vcf_pos]
                    alt_snp = alt_snps[idx]
                    allele_match = (
                        alt_snp == vcf_alt or
                        (len(alt_snp) > 1 and vcf_alt_freq >= min_freq_include) or
                        alt_snp in vcf_alt
                    )

                    if allele_match and vcf_depth >= min_total_depth and vcf_alt_freq >= min_freq_include:
                        updates[idx] = str(vcf_alt_freq)
                        recovered_complex += 1
                        logger.debug(
                            f"[RECAL] {sample} | complex pos {vcf_pos} | "
                            f"0 → {vcf_alt_freq} (DP={vcf_depth}, "
                            f"conservative recovery, freq >= {min_freq_include})"
                        )

                # ── Complex variant recovery ──────────────────────────────
                # FreeBayes tags complex rearrangements with 'complex' in
                # the INFO field. These can affect nearby positions within
                # a ±complex_window bp window, so we check all '0' positions
                # in that range and apply the same recovery thresholds.
                elif 'complex' in line:
                    vcf_pos_int = int(vcf_pos)
                    window_strs = {
                        str(p)
                        for p in range(
                            vcf_pos_int - complex_window,
                            vcf_pos_int + complex_window + 1
                        )
                    }

                    # Intersect window with positions pending recalibration
                    # (blocked positions already removed from zero_positions)
                    nearby = window_strs.intersection(zero_positions.keys())

                    for near_pos in nearby:
                        idx = zero_positions[near_pos]

                        if vcf_depth >= min_total_depth:
                            if vcf_alt_freq >= min_freq_discard:
                                updates[idx] = str(vcf_alt_freq)
                                recovered_numeric += 1
                                logger.debug(
                                    f"[RECAL] {sample} | complex pos {near_pos} "
                                    f"(vcf {vcf_pos}) | 0 → {vcf_alt_freq} "
                                    f"(DP={vcf_depth})"
                                )
                            else:
                                updates[idx] = '0'
                                logger.debug(
                                    f"[RECAL] {sample} | complex pos {near_pos} "
                                    f"| depth OK but freq < min_freq_discard → '0'"
                                )

                        elif vcf_depth > min_cov and vcf_alt_freq >= min_freq_include:
                            updates[idx] = '?'
                            recovered_lowfreq += 1
                            logger.debug(
                                f"[RECAL] {sample} | complex pos {near_pos} "
                                f"(vcf {vcf_pos}) | 0 → '?' (DP={vcf_depth})"
                            )

                        else:
                            updates[idx] = '!'
                            recovered_lowcov += 1
                            logger.debug(
                                f"[RECAL] {sample} | complex pos {near_pos} "
                                f"(vcf {vcf_pos}) | 0 → '!' (DP={vcf_depth})"
                            )

                    nearby_blocked = window_strs.intersection(blocked_zero_positions.keys())
                    for near_pos in nearby_blocked:
                        idx = blocked_zero_positions[near_pos]
                        if vcf_depth >= min_total_depth and vcf_alt_freq >= min_freq_include:
                            updates[idx] = str(vcf_alt_freq)
                            recovered_complex += 1
                            logger.debug(
                                f"[RECAL] {sample} | complex window pos {near_pos} "
                                f"(vcf {vcf_pos}) | 0 → {vcf_alt_freq} (conservative)"
                            )

    except Exception as exc:
        logger.warning(
            RED + BOLD + f"[RECAL] " + NORMAL + f"Error processing {sample}: {exc}" + END_FORMATTING
        )
        return sample, {}

    # ── Summary log per sample ────────────────────────────────────────────────
    if updates:
        logger.debug(
            f"[RECAL] {sample} | "
            f"recovered: {recovered_numeric} numeric, "
            f"{recovered_lowfreq} as '?', "
            f"{recovered_lowcov} as '!' "
            f"(total: {len(updates)})"
        )
    else:
        logger.debug(f"[RECAL] {sample} | no positions recovered")

    return sample, updates


def recalibrate_ddbb_vcf_intermediate(
    snp_matrix: pl.DataFrame,
    variant_folder: str,
    min_total_depth: int   = 12,
    min_cov: int           = 5,
    min_freq_include: float = 0.7,
    min_freq_discard: float = 0.2,
    complex_window: int    = 5,
    max_workers: int       = 1,
    complex_positions_per_sample: dict = None,
) -> pl.DataFrame:

    """
    Recalibrates the intermediate SNP matrix by cross-checking '0' cells
    against each sample's raw FreeBayes VCF.

    Replaces the legacy pandarallel + iterrows approach with:
      - ThreadPoolExecutor for parallel VCF reading across samples
      - Polars for all matrix operations (no Pandas dependency)

    Parameters
    ----------
    snp_matrix      : pl.DataFrame from ddbb_create_intermediate
    variant_folder  : root directory containing per-sample subdirectories
    min_total_depth : minimum depth for confident variant recovery
    min_cov         : minimum depth floor (below → '!')
    min_freq_include: minimum frequency for '?' assignment
    complex_window  : bp window around complex variants (default ±5)
    max_workers     : threads for parallel VCF processing
    complex_positions_per_sample : dict {sample: set_of_int_positions} of
                                   complex/hotspot positions to permanently block
                                   from recovery; None → no blocking

    Returns
    -------
    pl.DataFrame with recalibrated values and updated N and Samples columns
    """

    # ── Extract matrix structure ──────────────────────────────────────────────
    # All columns except Position, N, Samples are treated as sample columns.
    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in snp_matrix.columns if c not in index_cols]

    logger.info(
        GREEN + BOLD + f"[RECAL] " + NORMAL + f"Starting VCF recalibration for {len(sample_cols)} samples " + END_FORMATTING
    )

    logger.info(
        GREEN + BOLD + f"[RECAL] " + NORMAL + f"Matrix loaded: {snp_matrix.height} positions × "
        f"{len(sample_cols)} samples" + END_FORMATTING
    )

    # ── Parse position and allele info once for all samples ───────────────────
    # Position format: REGION|REF|POS|ALT
    # Extracting POS (index 2) and ALT (index 3) once avoids repeating
    # this string split inside every per-sample recalibration call.
    pos_df = snp_matrix.select('Position').with_columns([
        pl.col('Position').str.split('|').list.get(2).alias('POS'),
        pl.col('Position').str.split('|').list.get(3).alias('ALT'),
    ])

    positions = pos_df['POS'].to_list()   # aligned with matrix rows
    alt_snps  = pos_df['ALT'].to_list()   # aligned with matrix rows

    # ── Convert matrix to mutable dict of lists ───────────────────────────────
    # Polars Series are immutable, so we extract each sample column into a
    # plain Python list. Updates from recheck_variant_rawvcf_intermediate are
    # applied in-place by row index, which is O(1) per update.
    # The dict is then used to rebuild the Polars DataFrame at the end.
    matrix_dict: dict = {
        s: snp_matrix[s].to_list()
        for s in sample_cols
    }

    # ── Parallel VCF recalibration ────────────────────────────────────────────
    # Each sample's VCF is read independently, so parallelisation with
    # ThreadPoolExecutor is safe — no shared mutable state during reading.
    # Updates are collected per-sample and applied sequentially afterwards.

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:

        futures = {
            executor.submit(
                recheck_variant_rawvcf_intermediate,
                sample,
                variant_folder,
                positions,
                alt_snps,
                matrix_dict[sample],
                min_total_depth,
                min_cov,
                min_freq_include,
                min_freq_discard,
                complex_window,
                complex_positions_per_sample.get(sample, set())
                if complex_positions_per_sample else set(),
            ): sample
            for sample in sample_cols
        }

        total_updates = 0

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            sample, updates = result

            if updates:
                # Apply per-sample updates to the mutable matrix dict
                for idx, new_val in updates.items():
                    matrix_dict[sample][idx] = new_val
                total_updates += len(updates)

    # ── Rebuild Polars DataFrame from updated dict ────────────────────────────
    # Start from the index columns (Position, N, Samples) and re-attach
    # each sample column from the (now updated) matrix_dict.
    df = snp_matrix.select(index_cols)

    for s in sample_cols:
        df = df.with_columns(pl.Series(s, matrix_dict[s], dtype=pl.String))

    # ── Recompute N and Samples after recalibration ───────────────────────────
    # N and Samples must be recomputed because recovered positions change
    # which samples are considered 'present' at each variant position.
    # A cell is 'present' if its value is not '0', '!' or null.
    present_flags = [
        (~pl.col(s).is_in(['!', '0'])).alias(f'__p_{s}')
        for s in sample_cols
    ]
    df = df.with_columns(present_flags)

    flag_cols = [f'__p_{s}' for s in sample_cols]

    df = df.with_columns(
        pl.sum_horizontal(flag_cols).cast(pl.Int32).alias('N')
    )

    sample_names   = np.array(sample_cols)
    present_matrix = df.select(flag_cols).to_numpy().astype(bool)
    samples_col    = [",".join(sample_names[row]) for row in present_matrix]

    df = (
        df.with_columns(pl.Series('Samples', samples_col))
          .drop(flag_cols)
    )

    # Final column order: Position | N | Samples | <sample columns>
    df = df.select(index_cols + sample_cols)

    logger.info(GREEN + BOLD + "[RECAL] Recalibration complete. " + NORMAL + f"Total positions updated across all samples: {total_updates}" + END_FORMATTING + '\n')

    return df


def bed_to_df(bed_file: str) -> pl.DataFrame:

    """
    Reads a BED file (tab-separated, no header) into a Polars DataFrame.

    Automatically skips header lines by detecting whether columns 2 and 3
    (start, end) are numeric. Keeps only the first 4 columns.

    Expected BED format (no header):
        CHROM  start  end  strand/description

    Returns
    -------
    pl.DataFrame with columns: #CHROM | start | end | description
    """

    # ── Detect header lines ───────────────────────────────────────────────────
    header_lines = 0
    with open(bed_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 3 and fields[1].isdigit() and fields[2].isdigit():
                break
            header_lines += 1

    # ── Read with Polars ──────────────────────────────────────────────────────
    df = pl.read_csv(
        bed_file,
        separator='\t',
        has_header=False,
        skip_rows=header_lines,
        schema_overrides={
            'column_1': pl.String,
            'column_2': pl.Int32,
            'column_3': pl.Int32,
            'column_4': pl.String,
        },
        null_values=["", "NA"],
    )

    # Keep only first 4 columns and rename
    df = df.select(df.columns[:4])
    df.columns = ['#CHROM', 'start', 'end', 'description']

    return df


def remove_bed_positions(
    df: pl.DataFrame,
    bed_file: str,
    path_compare: str,
) -> pl.DataFrame:

    """
    Removes from df all variant positions that fall within any BED interval.

    Replaces the legacy row-by-row iterrows() approach with a fully
    vectorised Polars operation:
      1. Extract position numbers from the 'Position' column in one pass.
      2. Build a boolean mask checking each position against all BED
         intervals using a cross-join + is_between, which runs in Rust.
      3. Split, log, export filtered rows, and return the cleaned df.

    Parameters
    ----------
    df           : variant matrix from ddbb_create_intermediate (pl.DataFrame)
    bed_file     : path to the BED file with regions to exclude
    path_compare : base path for the output filtered-positions TSV

    Returns
    -------
    pl.DataFrame with BED-overlapping positions removed
    """

    filtered_position = path_compare + ".filter_position.tsv"

    # ── Load BED ──────────────────────────────────────────────────────────────
    bed_df = bed_to_df(bed_file)

    if bed_df.is_empty():
        logger.warning(
            RED + BOLD + f"[BED] " + NORMAL + f"Empty BED file: {bed_file}. No positions removed." + END_FORMATTING
        )
        return df

    # ── Extract numeric position from 'Position' column ──────────────────────
    # Format: REGION|REF|POS|ALT  →  extract field at index 2
    df = df.with_columns(
        pl.col('Position')
        .str.split('|')
        .list.get(2)
        .cast(pl.Int32)
        .alias('_pos_num')
    )

    # ── Vectorised interval lookup ────────────────────────────────────────────
    # For each position, check if it falls within any BED interval.
    # Cross-join produces all (position, interval) pairs; then we group by
    # position and flag those where any interval contains it.
    # This avoids any Python-level loop over rows.

    pos_series = df.select(['Position', '_pos_num']).unique('Position')

    # Cross join: every position against every BED interval
    hits = (
        pos_series
        .join(bed_df.select(['start', 'end']), how='cross')
        .filter(
            pl.col('_pos_num').is_between(
                pl.col('start'), pl.col('end'), closed='both'
            )
        )
        .select('Position')
        .unique()
    )

    # ── Split df into removed / kept ──────────────────────────────────────────
    removed = df.join(hits, on='Position', how='inner').drop('_pos_num')
    df    = df.join(hits, on='Position', how='anti').drop('_pos_num')

    # ── Log removed positions ─────────────────────────────────────────────────
    if removed.is_empty():
        logger.info(
            YELLOW + BOLD + f"[BED] " + NORMAL + f"No positions found in BED intervals: {bed_file}" + END_FORMATTING
        )
        return df

    for pos in removed['Position'].to_list():
        logger.debug(f"[BED] Position removed: {pos}")
        # logger.info(f"[BED] Position removed: {pos} (found in {bed_file})")

    logger.info(
        '\n' + BOLD + GREEN + f"[BED] " + NORMAL + f"{removed.height} position(s) removed from {bed_file}" + END_FORMATTING + '\n' +
        f"Filtered positions saved to: {filtered_position}" + END_FORMATTING + '\n'
    )

    # ── Export filtered positions ─────────────────────────────────────────────
    removed.write_csv(filtered_position, separator="\t")

    return df


def extract_bed_positions(
    df: Union[pl.DataFrame, str],
    bed_file: str,
    path_compare: str,
) -> pl.DataFrame:

    """
    Extracts and annotates variant positions that fall within any BED interval.

    Unlike remove_bed_positions (which removes matching positions), this
    function keeps only the matching positions and adds a 'description'
    column with the BED annotation for each match.

    Uses a vectorised cross-join approach instead of the legacy nested
    iterrows loop (O(n×m) → O(1) Polars operations).

    Parameters
    ----------
    df           : variant matrix from revised_df (pl.DataFrame or TSV path)
    bed_file     : path to the BED file with regions of interest
    path_compare : base path for the output annotated-positions TSV

    Returns
    -------
    pl.DataFrame with only the BED-overlapping positions and a
    'description' column added
    """

    output_file = path_compare + ".annotated_position.tsv"

    # ── Load BED ──────────────────────────────────────────────────────────────
    bed_df = bed_to_df(bed_file)

    if bed_df.is_empty():
        logger.warning(
            YELLOW + BOLD + f"[BED] " + NORMAL + f"Empty BED file: {bed_file}. No positions annotated." + END_FORMATTING + '\n'
        )
        return pl.DataFrame()

    # ── Extract numeric position from 'Position' column ──────────────────────
    # Format: REGION|REF|POS|ALT → extract field at index 2
    df = df.with_columns(
        pl.col('Position')
        .str.split('|')
        .list.get(2)
        .cast(pl.Int32)
        .alias('_pos_num')
    )

    # ── Vectorised interval lookup with annotation ────────────────────────────
    # Cross-join every position against every BED interval, filter matches,
    # and keep the description from the matching interval.
    # If a position matches multiple intervals, the first match is kept.
    pos_series = df.select(['Position', '_pos_num']).unique('Position')

    hits = (
        pos_series
        .join(bed_df.select(['start', 'end', 'description']), how='cross')
        .filter(
            pl.col('_pos_num').is_between(
                pl.col('start'), pl.col('end'), closed='both'
            )
        )
        .group_by('Position')
        .agg(pl.col('description').first())
    )

    if hits.is_empty():
        logger.info(YELLOW + BOLD + f"[BED] " + NORMAL + f"No positions found within BED intervals: {bed_file}" + END_FORMATTING +'\n')
        return pl.DataFrame()

    # ── Join annotation back onto matching rows ───────────────────────────────
    annotated = (
        df.join(hits, on='Position', how='inner')
          .drop('_pos_num')
    )

    # ── Log and export ────────────────────────────────────────────────────────
    for pos in annotated['Position'].to_list():
        logger.info(GREEN + BOLD + f"[BED] " + NORMAL + f"Position annotated: {pos} (found in {bed_file})" + END_FORMATTING)

    logger.info(
        '\n' + GREEN + BOLD + f"[BED] " + NORMAL + f"{annotated.height} position(s) annotated. "
        f"Saved to: {output_file}" + END_FORMATTING + '\n'
    )

    annotated.write_csv(output_file, separator='\t')

    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_hamming_distance(a: np.ndarray, b: np.ndarray) -> float:

    """
    Compute weighted Hamming distance between two float arrays.

    Encoding:
      0.0 vs 0.0 → 0.0  |  1.0 vs 1.0 → 0.0  |  0.5 vs 0.5 → 0.0
      0.0 vs 1.0 → 1.0  |  1.0 vs 0.0 → 1.0
      0.0 vs 0.5 → 0.5  |  0.5 vs 0.0 → 0.5
      1.0 vs 0.5 → 0.5  |  0.5 vs 1.0 → 0.5

    The distance is the sum of per-position weights, NOT normalised by
    the number of positions, so it represents actual SNP distance.
    """
    return float(np.sum(np.abs(a - b)))


def compute_distance_matrix(
    matrix: np.ndarray,
    sample_names: List[str],
    max_workers: int = 1,
) -> Tuple[np.ndarray, pl.DataFrame]:

    """
    Compute the full pairwise weighted Hamming distance matrix.

    Parameters
    ----------
    matrix       : float array of shape (n_positions, n_samples)
    sample_names : list of sample name strings
    max_workers  : threads for parallel row computation

    Returns
    -------
    dist_matrix : np.ndarray (n_samples, n_samples) of SNP distances
    pairwise_df : pl.DataFrame with columns sample_1 | sample_2 | distance
    """

    n = len(sample_names)
    # Transpose so rows = samples, columns = positions
    m = matrix.T  # shape (n_samples, n_positions)

    dist_matrix = np.zeros((n, n), dtype=float)

    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Computing pairwise distances for {n} samples..." + END_FORMATTING)

    def compute_row(i: int) -> Tuple[int, np.ndarray]:
        row = np.array([weighted_hamming_distance(m[i], m[j]) for j in range(n)])
        return i, row

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        for i, row in executor.map(compute_row, range(n)):
            dist_matrix[i] = row

    logger.info(GREEN + BOLD + "[COMPARE] " + NORMAL + "Distance matrix computed." + END_FORMATTING)

    # Build pairwise DataFrame
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((sample_names[i], sample_names[j], dist_matrix[i, j]))

    pairwise_df = pl.DataFrame({
        'sample_1': [r[0] for r in rows],
        'sample_2': [r[1] for r in rows],
        'distance': [r[2] for r in rows],
    })

    return dist_matrix, pairwise_df


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_clusters(
    dist_matrix: np.ndarray,
    sample_names: List[str],
    distance_threshold: int,
) -> pl.DataFrame:

    """
    Assign samples to clusters based on distance threshold using
    single-linkage connected components (two samples are in the same
    cluster if their distance <= threshold).

    Returns pl.DataFrame: sample | cluster_id | cluster_size
    """

    n = len(sample_names)
    # Union-Find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= distance_threshold:
                union(i, j)

    # Group by root
    groups: Dict[int, List[str]] = defaultdict(list)
    for i, name in enumerate(sample_names):
        groups[find(i)].append(name)

    # Sort clusters by size descending and reassign sequential IDs
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    rows = []
    for cluster_id, members in enumerate(sorted_groups, 1):
        for member in members:
            rows.append((member, cluster_id, len(members)))

    return pl.DataFrame({
        'sample'      : [r[0] for r in rows],
        'cluster_id'  : [r[1] for r in rows],
        'cluster_size': [r[2] for r in rows],
    }).sort('cluster_id')


# ═══════════════════════════════════════════════════════════════════════════════
# Final graphic section
# ═══════════════════════════════════════════════════════════════════════════════

# ── Font size for sample labels in dendrogram ─────────────────────────────────
# change size: adjust LABEL_FONT_SIZE (10-12 recommended)
LABEL_FONT_SIZE = 11

# ── Progressive rendering threshold ───────────────────────────────────────────
# Maximum number of leaf labels rendered at global zoom level.
# Sub-branches with more leaves than this are collapsed into summary nodes.
MAX_VISIBLE_LEAVES = 200

MJ_THRESHOLD = 50

# ═══════════════════════════════════════════════════════════════════════════════
# SNP EXCLUSIVITY + ANCESTRAL TRACING
# ═══════════════════════════════════════════════════════════════════════════════

def _get_all_leaves(node) -> List[int]:
    """Recursively collect all leaf sample indices under a scipy ClusterNode."""
    if node.is_leaf():
        return [node.id]
    return _get_all_leaves(node.left) + _get_all_leaves(node.right)


def _assign_clusters_local(
    dist_matrix: np.ndarray,
    n: int,
    threshold: float,
) -> Dict[int, List[int]]:
    """Union-Find cluster assignment. Returns {root: [member_indices]}."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= threshold:
                rx, ry = find(i), find(j)
                if rx != ry:
                    parent[rx] = ry

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return dict(groups)


def compute_cluster_snps(
    snp_df: pl.DataFrame,
    dist_matrix: np.ndarray,
    sample_names: List[str],
    linkage_matrix: np.ndarray,
    distance_threshold: int,
    output_file: str,
) -> Tuple[pl.DataFrame, Dict[int, List[str]]]:

    """
    Compute common and unique SNPs per cluster, plus ancestral tracing.

    Returns
    -------
    cluster_df : pl.DataFrame with columns:
                   cluster_id | n_samples | n_common_snps | positions |
                   n_unique_snps | unique_positions
    node_snps  : dict {postorder_node_index: [exclusive_position_strings]}
                 for use in the dendrogram node inspector panel
    """

    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in snp_df.columns if c not in index_cols]
    positions   = snp_df['Position'].to_list()
    n           = len(sample_names)

    # Build float matrix (n_positions × n_samples)
    raw          = snp_df.select(sample_cols).to_numpy()
    float_matrix = np.zeros(raw.shape, dtype=float)
    for j in range(raw.shape[1]):
        for i in range(raw.shape[0]):
            try:
                float_matrix[i, j] = float(raw[i, j])
            except (ValueError, TypeError):
                float_matrix[i, j] = 0.0

    matrix_T = float_matrix.T  # (n_samples × n_positions)

    # ── Cluster assignment ────────────────────────────────────────────────────
    groups        = _assign_clusters_local(dist_matrix, n, distance_threshold)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    idx_to_cid: Dict[int, int]       = {}
    cid_members: Dict[int, List[int]] = {}
    for cid, members in enumerate(sorted_groups):
        cid_members[cid] = members
        for m in members:
            idx_to_cid[m] = cid

    all_cids = list(cid_members.keys())

    # ── Common SNPs per cluster ───────────────────────────────────────────────
    cid_common: Dict[int, Set[int]] = {}
    for cid, members in cid_members.items():
        if not members:
            cid_common[cid] = set()
            continue
        mask = np.all(matrix_T[members, :] >= 0.5, axis=0)
        cid_common[cid] = set(np.where(mask)[0].tolist())

    # ── Unique SNPs per cluster ───────────────────────────────────────────────
    cid_unique: Dict[int, Set[int]] = {}
    for cid in all_cids:
        others: Set[int] = set()
        for other in all_cids:
            if other != cid:
                others |= cid_common[other]
        cid_unique[cid] = cid_common[cid] - others

    # ── Ancestral tracing (post-order, bottom-up) ─────────────────────────────
    tree               = shc.to_tree(linkage_matrix, False)
    assigned: Set[int] = set()
    node_snps: Dict[int, List[str]] = {}

    def get_desc_cids(node) -> Set[int]:
        leaves = set(_get_all_leaves(node))
        valid_cids = set()

        for cid, members in cid_members.items():
            if set(members).issubset(leaves):
                valid_cids.add(cid)

        return valid_cids

    def traverse(node):
        if node.is_leaf():
            return
        traverse(node.left)
        traverse(node.right)

        nidx = node.id
        desc      = get_desc_cids(node)
        non_desc  = set(all_cids) - desc

        if not desc or not non_desc:
            node_snps[nidx] = []
            return

        # Positions shared by ALL descendant clusters
        shared = set.intersection(*[cid_common[c] for c in desc])

        # Positions absent in ALL non-descendant cluster members
        nd_members = []
        for c in non_desc:
            nd_members.extend(cid_members[c])

        if nd_members:
            absent_mask = np.all(matrix_T[nd_members, :] < 0.5, axis=0)
            absent_idx  = set(np.where(absent_mask)[0].tolist())
        else:
            absent_idx = set(range(len(positions)))

        # Exclusive to this node, not yet assigned to any descendant
        excl = (shared & absent_idx) - assigned
        assigned.update(excl)
        node_snps[nidx] = [positions[i] for i in sorted(excl)]

    traverse(tree)

    # ── Build output DataFrame ────────────────────────────────────────────────
    rows = []
    for cid in sorted(all_cids):
        common_idx = sorted(cid_common[cid])
        unique_idx = sorted(cid_unique[cid])
        rows.append({
            'cluster_id'      : cid + 1,
            'n_samples'       : len(cid_members[cid]),
            'n_common_snps'   : len(common_idx),
            'positions'       : '; '.join(positions[i] for i in common_idx),
            'n_unique_snps'   : len(unique_idx),
            'unique_positions': '; '.join(positions[i] for i in unique_idx),
        })

    cluster_df = pl.DataFrame(rows)
    cluster_df.write_csv(output_file, separator='\t')
    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Cluster SNPs saved: {output_file}" + END_FORMATTING + '\n')

    return cluster_df, node_snps


# ═══════════════════════════════════════════════════════════════════════════════
# DENDROGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dendrogram(
    linkage_matrix: np.ndarray,
    sample_names: List[str],
    node_snps: Dict[int, List[str]],
    dist_matrix: np.ndarray,
    distance_threshold: int,
    output_file: str,
) -> None:

    """
    Interactive dendrogram — Horizontal (root←leaves) and Vertical (root↑leaves).

    Mapping strategy (v5):
    • SNPs  → node_snps.get(si)      segment index, matches compute_cluster_snps key
    • Members → leaf_sets[n+seg2row[si]]  correct linkage-row via scipy position formula
    Both are independent lookups, each using the right key for its source.

    Other fixes vs v4:
    • BR = MAX_D * 0.002 (small badges)
    • Fit-to-content on open via viewBox + preserveAspectRatio
    • Scale bar in data-space
    • Click active node again → deselects
    • Bottom margin enlarged (BOT_PAD = ROW_H * 4)
    """

    import json as _json, math as _math

    n     = len(sample_names)
    max_d = float(np.max(dist_matrix)) or 1.0

    # ── Mode detection ────────────────────────────────────────────────────────
    _par = list(range(n))
    def _find(x):
        while _par[x] != x: _par[x] = _par[_par[x]]; x = _par[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= distance_threshold:
                a, b = _find(i), _find(j)
                if a != b: _par[a] = b
    mode_a = len({_find(i) for i in range(n)}) == 1

    fig_tmp, ax = plt.subplots(figsize=(1, 1))
    ddata = shc.dendrogram(linkage_matrix, labels=sample_names,
                           no_plot=True, color_threshold=distance_threshold)
    plt.close(fig_tmp)

    icoord = ddata['icoord']
    dcoord = ddata['dcoord']
    ivl    = ddata['ivl']

    # ── Leaf-set per node ─────────────────────────────────────────────────────
    leaf_sets: Dict[int, frozenset] = {i: frozenset([i]) for i in range(n)}
    for i, row in enumerate(linkage_matrix):
        l, r = int(row[0]), int(row[1])
        leaf_sets[n + i] = leaf_sets[l] | leaf_sets[r]

    # ── seg → linkage-row mapping (for members) ───────────────────────────────
    # Reproduce scipy's position formula: pos[internal] = mean(pos[left], pos[right])
    # Then match each segment's centre (ic[1]+ic[2])/2 to its linkage row.
    name2rank = {nm: rank for rank, nm in enumerate(ivl)}
    orig2rank  = {orig: name2rank[sample_names[orig]] for orig in range(n)}
    node_pos: Dict[int, float] = {i: orig2rank[i] * 10 + 5.0 for i in range(n)}
    for i, row in enumerate(linkage_matrix):
        l, r = int(row[0]), int(row[1])
        node_pos[n + i] = (node_pos[l] + node_pos[r]) / 2.0
    pos2node = {round(node_pos[n + i], 6): n + i
                for i in range(len(linkage_matrix))}

    seg2row: Dict[int, int] = {}
    for si, ic in enumerate(icoord):
        centre = round((ic[1] + ic[2]) / 2.0, 6)
        nid    = pos2node.get(centre)
        if nid is not None:
            seg2row[si] = nid - n

    # ── Cluster colour per leaf ───────────────────────────────────────────────
    palette = [
        '#1976D2','#388E3C','#D32F2F','#F57C00',
        '#7B1FA2','#0097A7','#689F38','#C2185B',
        '#5D4037','#455A64','#AFB42B','#E64A19',
    ]
    leaf_cluster: Dict[int, int] = {i: i for i in range(n)}
    for i, row in enumerate(linkage_matrix):
        if row[2] <= distance_threshold:
            nid = n + i
            for leaf in leaf_sets[nid]:
                leaf_cluster[leaf] = nid
    cluster_roots  = sorted(set(leaf_cluster.values()))
    cluster_colour = {cid: palette[j % len(palette)]
                      for j, cid in enumerate(cluster_roots)}
    INTER_COL = '#90a4ae'

    def _seg_colour(si: int) -> str:
        row_i = seg2row.get(si)
        if row_i is None:
            return INTER_COL
        cids = {leaf_cluster[l] for l in leaf_sets[n + row_i]}
        return cluster_colour[next(iter(cids))] if len(cids) == 1 else INTER_COL

    # ── SNP formatter ─────────────────────────────────────────────────────────
    def _fmt(snps):
        out = []
        for s in snps[:120]:
            p = s.split('|')
            out.append(f"{p[0]} · {p[2]} {p[1]}→{p[3]}" if len(p) >= 4 else s)
        return out

    # ── Scale bar ─────────────────────────────────────────────────────────────
    def _nice_scale(md):
        raw = md * 0.15
        mag = 10 ** _math.floor(_math.log10(max(raw, 1)))
        for m in [1, 2, 5, 10]:
            if raw <= m * mag:
                return m * mag
        return mag * 10

    scale_len = _nice_scale(max_d)

    # ── Data-space sizing ─────────────────────────────────────────────────────
    ROW_H    = 10
    FONT     = 5.5
    CHAR_W   = FONT * 0.60
    max_lbl  = max((len(nm) for nm in sample_names), default=4)
    LBL_W    = max_lbl * CHAR_W + 6
    LBL_H    = max_lbl * CHAR_W + 6
    ROOT_PAD = max_d * 0.04
    BOT_PAD  = ROW_H * 4          # enlarged bottom margin

    H_VBX = -ROOT_PAD;  H_VBY = -ROW_H * 0.5
    H_VBW = max_d + LBL_W + ROOT_PAD
    H_VBH = n * ROW_H + ROW_H * 0.5 + BOT_PAD

    V_VBX = -ROW_H * 0.5;  V_VBY = -ROOT_PAD
    V_VBW = n * ROW_H + ROW_H
    V_VBH = max_d + LBL_H + ROOT_PAD + BOT_PAD

    H_SCALE_X = 0.0
    H_SCALE_Y = n * ROW_H + ROW_H * 1.5
    V_SCALE_X = 0.0
    V_SCALE_Y = max_d + LBL_H * 0.5 + ROOT_PAD + ROW_H

    # ── Build JS segment data ─────────────────────────────────────────────────
    # SNPs  → node_snps.get(si)          key = segment draw-order index
    # Members → leaf_sets[n+seg2row[si]] key = linkage row via position formula
    segs_js = []
    for si, (ic, dc) in enumerate(zip(icoord, dcoord)):
        row_i = seg2row.get(si)

        if row_i is not None:
            nid = n + row_i
            snps_raw = node_snps.get(nid, []) 
            members  = sorted(sample_names[l] for l in leaf_sets[nid])
        else:
            snps_raw = []
            members  = []

        snps_fmt = _fmt(snps_raw)
        jic = (ic[1] + ic[2]) / 2.0
        jdc = dc[1]

        segs_js.append({
            'hx'     : [max_d - v for v in dc],
            'hy'     : list(ic),
            'hjx'    : max_d - jdc,
            'hjy'    : jic,
            'vx'     : list(ic),
            'vy'     : [max_d - v for v in dc],
            'vjx'    : jic,
            'vjy'    : max_d - jdc,
            'color'  : _seg_colour(si),
            'n_snp'  : len(snps_raw),
            'snps'   : snps_fmt,
            'members': members,
        })

    leaves_js = [{'name': nm, 'lp': rank * ROW_H + ROW_H / 2}
                 for rank, nm in enumerate(ivl)]

    mode_badge = 'Mode A — Single cluster' if mode_a else 'Mode B — Multiple clusters'
    HL = '#eeee0c'

    segs_json   = _json.dumps(segs_js)
    leaves_json = _json.dumps(leaves_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SNP Dendrogram</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;font-family:'Courier New',monospace;color:#37474f;
      display:flex;flex-direction:column;height:100vh;overflow:hidden}}
#tb{{height:46px;background:#f5f5f5;border-bottom:1px solid #ddd;
     display:flex;align-items:center;padding:0 14px;gap:8px;flex-shrink:0}}
#tb h1{{font-size:11px;color:#1565C0;letter-spacing:2px;
        text-transform:uppercase;font-weight:bold;white-space:nowrap}}
.sep{{width:1px;height:20px;background:#ddd;flex-shrink:0}}
button{{background:#fff;color:#37474f;border:1px solid #bdbdbd;border-radius:4px;
        padding:4px 10px;font-family:'Courier New',monospace;font-size:11px;
        cursor:pointer;transition:all .12s;white-space:nowrap}}
button:hover{{background:#e8f5e9;border-color:#2e7d32;color:#2e7d32}}
button.on{{background:#1565C0;color:#fff;border-color:#1565C0}}
#srch{{border:1px solid #bdbdbd;border-radius:4px;padding:4px 8px;
       font-family:'Courier New',monospace;font-size:11px;width:160px;outline:none}}
#srch:focus{{border-color:#1565C0}}
.bdg{{font-size:10px;background:#e8f5e9;color:#2e7d32;
      border:1px solid #a5d6a7;border-radius:3px;padding:2px 7px;white-space:nowrap}}
#flex{{flex:1}}
#main{{display:flex;flex:1;min-height:0}}
#wrap{{flex:1;min-width:0;overflow:hidden;cursor:grab;background:#fff}}
#wrap:active{{cursor:grabbing}}
#dend{{display:block;width:100%;height:100%}}
#panel{{width:280px;background:#fafafa;border-left:1px solid #e0e0e0;
        display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}}
#ph{{padding:12px 14px 8px;border-bottom:1px solid #e0e0e0;flex-shrink:0}}
#ph h2{{font-size:11px;color:#1565C0;text-transform:uppercase;letter-spacing:1px}}
#pb{{flex:1;overflow-y:auto;padding:10px 14px;font-size:11px;line-height:1.9}}
.sr{{display:flex;justify-content:space-between;color:#78909c}}
.sv{{color:#263238;font-weight:bold}}
.lbl{{font-size:10px;color:#b0bec5;text-transform:uppercase;
      letter-spacing:1px;margin:8px 0 3px}}
.hlist{{font-size:10px;max-height:200px;overflow-y:auto;
        background:#fff;border:1px solid #e0e0e0;border-radius:4px;
        padding:5px 7px;line-height:1.85}}
.snp-e{{color:#1565C0}}
.mbr-e{{color:#546e7a}}
#hint{{color:#b0bec5;font-size:11px;text-align:center;
       margin-top:40px;line-height:2.4}}
.branch{{fill:none;stroke-linecap:round;stroke-linejoin:round}}
.leaf-lbl{{font-family:'Courier New',monospace;fill:#546e7a;
           dominant-baseline:central;cursor:default}}
.leaf-lbl.hit{{fill:#D32F2F;font-weight:bold}}
.bc{{cursor:pointer}}
.bc.has{{fill:#1976D2;stroke:#0d47a1;stroke-width:0.3}}
.bc.none{{fill:#e0e0e0;stroke:#bdbdbd;stroke-width:0.2}}
.bc.active{{fill:{HL};stroke:#aaaa00;stroke-width:0.5}}
.bc:hover{{opacity:0.78}}
.bt{{font-family:'Courier New',monospace;fill:#fff;
     text-anchor:middle;dominant-baseline:central;pointer-events:none}}
.scale-line{{stroke:#546e7a;stroke-width:0.5;fill:none}}
.scale-txt{{font-family:'Courier New',monospace;fill:#546e7a;
            text-anchor:start;dominant-baseline:hanging}}
</style>
</head>
<body>
<div id="tb">
  <h1>Dendrogram</h1>
  <span class="bdg">{mode_badge}</span>
  <div class="sep"></div>
  <button id="bh" class="on" onclick="setLayout('h')">Root ← Leaves</button>
  <button id="bv"            onclick="setLayout('v')">Root ↑ Leaves</button>
  <div class="sep"></div>
  <input id="srch" type="text" placeholder="Search sample…" oninput="doSearch(this.value)">
  <div id="flex"></div>
  <button onclick="dlSVG()">Download SVG</button>
</div>
<div id="main">
  <div id="wrap">
    <svg id="dend" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMinYMin meet">
      <g id="zg">
        <g id="bg"></g>
        <g id="ng"></g>
        <g id="lg"></g>
        <g id="sg"></g>
      </g>
    </svg>
  </div>
  <div id="panel">
    <div id="ph"><h2>Node Inspector</h2></div>
    <div id="pb">
      <div class="sr"><span>Total samples</span><span class="sv">{n}</span></div>
      <div class="sr"><span>Distance threshold</span>
           <span class="sv">{distance_threshold} SNPs</span></div>
      <div id="hint">Click a branch node<br>to inspect SNPs and<br>involved samples</div>
      <div id="nd" style="display:none">
        <div class="lbl" id="snp-lbl">Exclusive SNPs</div>
        <div class="hlist" id="snp-list"></div>
        <div class="lbl" id="mbr-lbl">Samples in subtree</div>
        <div class="hlist" id="mbr-list"></div>
        <div style="margin-top:8px">
          <button onclick="dlMembers()" style="width:100%;font-size:10px">
            ↓ Download sample list
          </button>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
var SEGS    = {segs_json};
var LEAVES  = {leaves_json};
var MAX_D   = {max_d:.6f};
var N       = {n};
var FONT    = {FONT:.2f};
var H_VBX={H_VBX:.4f}, H_VBY={H_VBY:.4f}, H_VBW={H_VBW:.4f}, H_VBH={H_VBH:.4f};
var V_VBX={V_VBX:.4f}, V_VBY={V_VBY:.4f}, V_VBW={V_VBW:.4f}, V_VBH={V_VBH:.4f};
var SCALE_LEN={scale_len};
var H_SCALE_X={H_SCALE_X:.4f}, H_SCALE_Y={H_SCALE_Y:.4f};
var V_SCALE_X={V_SCALE_X:.4f}, V_SCALE_Y={V_SCALE_Y:.4f};

var svg = d3.select('#dend');
var zg  = d3.select('#zg');
var bg  = d3.select('#bg');
var ng  = d3.select('#ng');
var lg  = d3.select('#lg');
var sg  = d3.select('#sg');
var layout    = 'h';
var activeIdx = -1;
var _dlData   = null;

var zoom = d3.zoom().scaleExtent([0.005, 120])
  .on('zoom', function(e) {{ zg.attr('transform', e.transform); }});
svg.call(zoom);

/* ── Panel ── */
function showPanel(idx) {{
  /* click same node → deselect */
  if (activeIdx === idx) {{
    activeIdx = -1;
    _dlData   = null;
    ng.selectAll('.bc').classed('active', false);
    document.getElementById('nd').style.display   = 'none';
    document.getElementById('hint').style.display = 'block';
    return;
  }}
  activeIdx = idx;
  var s = SEGS[idx];
  ng.selectAll('.bc').classed('active', false);
  ng.selectAll('.bc').filter(function(d) {{ return d === idx; }}).classed('active', true);

  document.getElementById('hint').style.display = 'none';
  document.getElementById('nd').style.display   = 'block';

  document.getElementById('snp-lbl').textContent =
    s.n_snp + ' exclusive SNP' + (s.n_snp !== 1 ? 's' : '') + ' at this node';
  document.getElementById('snp-list').innerHTML = s.snps && s.snps.length
    ? s.snps.map(function(t) {{ return '<span class="snp-e">'+t+'</span>'; }}).join('<br>')
    : '<span style="color:#b0bec5">No exclusive SNPs at this node</span>';

  var mbrs = s.members || [];
  document.getElementById('mbr-lbl').textContent =
    mbrs.length + ' sample' + (mbrs.length !== 1 ? 's' : '') + ' in this subtree';
  document.getElementById('mbr-list').innerHTML = mbrs.length
    ? mbrs.slice(0,300).map(function(m) {{ return '<span class="mbr-e">'+m+'</span>'; }}).join('<br>') +
      (mbrs.length > 300 ? '<br><span style="color:#b0bec5">…and '+(mbrs.length-300)+' more</span>' : '')
    : '<span style="color:#b0bec5">—</span>';
  _dlData = mbrs;
}}

function dlMembers() {{
  if (!_dlData || !_dlData.length) return;
  var b = new Blob([_dlData.join('\\n')], {{type:'text/plain'}});
  var u = URL.createObjectURL(b);
  var a = document.createElement('a');
  a.href=u; a.download='subtree_members.txt'; a.click(); URL.revokeObjectURL(u);
}}

/* ── Search ── */
function doSearch(q) {{
  q = q.trim().toLowerCase();
  lg.selectAll('.leaf-lbl').classed('hit', function(d) {{
    return q.length > 0 && d.name.toLowerCase().indexOf(q) >= 0;
  }});
}}

/* ── Render ── */
function render(lay) {{
  bg.selectAll('*').remove();
  ng.selectAll('*').remove();
  lg.selectAll('*').remove();
  sg.selectAll('*').remove();
  activeIdx = -1; _dlData = null;
  document.getElementById('nd').style.display   = 'none';
  document.getElementById('hint').style.display = 'block';

  var isH = lay === 'h';
  var vb   = isH ? [H_VBX,H_VBY,H_VBW,H_VBH].join(' ')
                 : [V_VBX,V_VBY,V_VBW,V_VBH].join(' ');
  svg.attr('viewBox', vb);

  var line = d3.line().x(function(p){{return p[0];}})
                      .y(function(p){{return p[1];}}).curve(d3.curveLinear);

  /* Branches */
  SEGS.forEach(function(s) {{
    var pts = isH ? s.hx.map(function(x,k){{return [x,s.hy[k]];}})
                  : s.vx.map(function(x,k){{return [x,s.vy[k]];}});
    bg.append('path').attr('class','branch')
      .attr('d',line(pts)).attr('stroke',s.color).attr('stroke-width',0.8);
  }});

  /* SNP badges — BR = MAX_D * 0.002 */
  var BR = MAX_D * 0.002;
  SEGS.forEach(function(s,i) {{
    var bx = isH ? s.hjx : s.vjx;
    var by = isH ? s.hjy : s.vjy;
    var has = s.n_snp > 0;
    ng.append('circle').datum(i)
      .attr('class','bc '+(has?'has':'none'))
      .attr('cx',bx).attr('cy',by)
      .attr('r', has ? BR : BR*0.45)
      .on('click', function(){{ showPanel(i); }});
    if (has) {{
      ng.append('text').attr('class','bt')
        .attr('x',bx).attr('y',by)
        .attr('font-size', BR*0.85)
        .attr('pointer-events','none')
        .text(s.n_snp > 99 ? '99+' : s.n_snp);
    }}
  }});

  /* Leaf labels */
  LEAVES.forEach(function(lf) {{
    var x, y, rot;
    if (isH) {{ x=MAX_D+2; y=lf.lp; rot=null; }}
    else     {{ x=lf.lp;   y=MAX_D+2; rot='rotate(-55,'+x+','+y+')'; }}
    lg.append('text').datum(lf)
      .attr('class','leaf-lbl')
      .attr('x',x).attr('y',y)
      .attr('font-size',FONT)
      .attr('text-anchor','start')
      .attr('transform',rot)
      .text(lf.name);
  }});

  /* Scale bar */
  var sx = isH ? H_SCALE_X : V_SCALE_X;
  var sy = isH ? H_SCALE_Y : V_SCALE_Y;
  var tk = FONT * 0.7;
  sg.append('line').attr('class','scale-line')
    .attr('x1',sx).attr('y1',sy).attr('x2',sx+SCALE_LEN).attr('y2',sy);
  sg.append('line').attr('class','scale-line')
    .attr('x1',sx).attr('y1',sy-tk).attr('x2',sx).attr('y2',sy+tk);
  sg.append('line').attr('class','scale-line')
    .attr('x1',sx+SCALE_LEN).attr('y1',sy-tk)
    .attr('x2',sx+SCALE_LEN).attr('y2',sy+tk);
  sg.append('text').attr('class','scale-txt')
    .attr('x',sx).attr('y',sy+tk+1)
    .attr('font-size',FONT*0.85)
    .text(SCALE_LEN+' SNPs');

  /* Fit entire tree on open — reset zoom to identity so viewBox fills svg */
  svg.call(zoom.transform, d3.zoomIdentity);
}}

function setLayout(lay) {{
  layout = lay;
  document.getElementById('bh').classList.toggle('on', lay==='h');
  document.getElementById('bv').classList.toggle('on', lay==='v');
  render(lay);
}}

function dlSVG() {{
  var clone = document.getElementById('dend').cloneNode(true);
  var b = new Blob([new XMLSerializer().serializeToString(clone)],{{type:'image/svg+xml'}});
  var u = URL.createObjectURL(b);
  var a = document.createElement('a');
  a.href=u; a.download='dendrogram.svg'; a.click(); URL.revokeObjectURL(u);
}}

window.addEventListener('resize', function(){{ render(layout); }});
setLayout('h');
</script>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    mode_str = 'Mode A (Single cluster)' if mode_a else 'Mode B (Overview)'
    logger.info(
        GREEN + BOLD + f"[COMPARE] " + NORMAL +
        f"Dendrogram saved ({mode_str}): {output_file}" +
        END_FORMATTING
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GENOMIC NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

"""
Genomic network with two display modes:

Mode A — Median Joining (single cluster):
    All samples belong to the same cluster (max pairwise distance
    <= collapse_threshold). Renders a full Median Joining Network
    styled like PopART: real samples as circles, median vectors (MVs)
    as small diamonds, edge length proportional to SNP distance.
 
Mode B — Static overview (multiple clusters):
    Samples span more than one cluster. Renders a static force layout
    (positions pre-computed in Python, no D3 physics) for instant
    display even with 2000+ samples.
    - Cluster nodes: red, size proportional to member count
    - Singleton nodes: grey, small
    - No edges between nodes (simplified)
    - Zoom / pan enabled
    - Tooltip shows cluster members on hover
"""

# ─── helpers shared with the rest of the module ──────────────────────────────
def _hkey(h: np.ndarray) -> str:
    return h.tobytes().hex()

def _ham(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def _median(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).astype(np.int8)

def _prim(haps: Dict[str, np.ndarray], ids: List[str]) -> List[Tuple[str, str, int]]:
    if len(ids) < 2:
        return []

    intree    = {ids[0]}
    remaining = set(ids[1:])
    edges: List[Tuple[str, str, int]] = []

    while remaining:
        best = None
        for u in intree:
            for v in remaining:
                d = _ham(haps[u], haps[v])
                if best is None or d < best[2]:
                    best = (u, v, d)
        if best:
            edges.append(best)
            intree.add(best[1])
            remaining.discard(best[1])
    return edges

def _node_size_b(sz: int) -> float:
    """
    Fixed-tier node sizes for Mode B.
    singleton : 7
    2–5       : 14  (2 × singleton)
    6–10      : 21  (1.5 × previous tier)
    > 10      : 32  (1.5 × previous tier)
    """
    if sz == 1:    return 7.0
    elif sz <= 5:  return 14.0
    elif sz <= 10: return 21.0
    else:          return 32.0


# ─── Median Joining (unchanged logic, kept here for completeness) ─────────────
def _mj(names: List[str], mat: np.ndarray) -> Tuple[List[dict], List[dict]]:
    hap_to_names: Dict[str, List[str]] = defaultdict(list)
    hap_arr: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        k = _hkey(mat[i])
        hap_to_names[k].append(name)
        hap_arr[k] = mat[i].copy()

    node_ids = list(hap_arr.keys())
    mv_set: Set[str] = set()

    for _ in range(20):
        added = False
        mst   = _prim(hap_arr, node_ids)
        linked = list({nd for e in mst for nd in e[:2]})
        for i in range(len(linked)):
            for j in range(i + 1, len(linked)):
                for k in range(j + 1, len(linked)):
                    mv = _median(hap_arr[linked[i]],
                                 hap_arr[linked[j]],
                                 hap_arr[linked[k]])
                    mk = _hkey(mv)
                    if mk not in hap_arr:
                        hap_arr[mk]      = mv
                        hap_to_names[mk] = []
                        node_ids.append(mk)
                        mv_set.add(mk)
                        added = True
        if not added:
            break

    final = _prim(hap_arr, node_ids)

    deg: Dict[str, int] = defaultdict(int)
    for a, b, _ in final:
        deg[a] += 1; deg[b] += 1

    removable: Set[str] = set()
    for k in mv_set:
        if deg.get(k, 0) <= 2:
            nb = [(b if a == k else a, d)
                  for a, b, d in final if a == k or b == k]
            if len(nb) == 2:
                dd = _ham(hap_arr[nb[0][0]], hap_arr[nb[1][0]])
                if dd == nb[0][1] + nb[1][1]:
                    removable.add(k)

    node_ids = [k for k in node_ids if k not in removable]
    hap_arr  = {k: v for k, v in hap_arr.items() if k not in removable}
    mv_set  -= removable
    final    = _prim(hap_arr, node_ids)

    kid2int: Dict[str, int] = {k: i for i, k in enumerate(node_ids)}
    max_d = max((d for _, _, d in final), default=1) or 1

    nodes = []
    for k in node_ids:
        sn    = hap_to_names.get(k, [])
        is_mv = k in mv_set
        nodes.append({
            'id'       : kid2int[k],
            'label'    : '(MV)' if is_mv else ' / '.join(sn),
            'title'    : 'Hypothetical ancestor (MV)' if is_mv else '\n'.join(sn),
            'is_mv'    : is_mv,
            'color'    : '#b0bec5' if is_mv else '#1976D2',
            'stroke'   : '#78909c' if is_mv else '#0d47a1',
            'size'     : 6 if is_mv else max(10, min(10 + len(sn) * 3, 30)),
            'singleton': False,
            'members'  : sn,
        })

    seen: Set[Tuple[int, int]] = set()
    edges = []
    for a, b, d in final:
        ia, ib = kid2int[a], kid2int[b]
        p = (min(ia, ib), max(ia, ib))
        if p in seen:
            continue
        seen.add(p)
        edges.append({
            'source': ia, 'target': ib,
            'dist'  : d,
            'label' : str(d),
            'width' : max(1.2, 1.5 + 4.5 * d / max_d),
        })

    return nodes, edges


# ─── Mode A: MDS layout ───────────────────────────────────────────────────────
def _tree_layout_rooted(
    nodes: List[dict],
    edges: List[dict],
    ancestor_id: int,
    canvas_w: float = 900.0,
    canvas_h: float = 700.0,
    padding: float  = 80.0,
) -> List[dict]:

    """
    Left-to-right rooted tree layout for Median Joining networks (Mode A).

    * x-axis = cumulative SNP distance from the ancestor (root → left edge).
    * y-axis = equal-area branch allocation: each branch receives vertical
               space proportional to the number of leaves below it.

    Also writes ``target_len`` (pixels) onto every edge for the JS spring
    relaxer, so that branch lengths stay proportional to SNP distance after
    interactive dragging.
    """

    n = len(nodes)
    if n == 0:
        return nodes
    if n == 1:
        nodes[0]['x'] = canvas_w / 2.0
        nodes[0]['y'] = canvas_h / 2.0
        return nodes

    # ── Build undirected adjacency ────────────────────────────────────────────
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for e in edges:
        s, t, d = e['source'], e['target'], e.get('dist', 1)
        adj[s].append((t, int(d)))
        adj[t].append((s, int(d)))

    all_ids: Set[int] = {nd['id'] for nd in nodes}

    # ── BFS from ancestor: directed tree + cumulative SNP depth ──────────────
    visited:     Set[int]              = {ancestor_id}
    children_of: Dict[int, List[int]] = defaultdict(list)
    depth_snp:   Dict[int, float]     = {ancestor_id: 0.0}
    queue = [ancestor_id]

    while queue:
        curr = queue.pop(0)
        for nb, d in adj[curr]:
            if nb not in visited and nb in all_ids:
                visited.add(nb)
                children_of[curr].append(nb)
                depth_snp[nb] = depth_snp[curr] + float(d)
                queue.append(nb)

    # Safety: nodes disconnected from ancestor (shouldn't happen in MJ graph)
    for nd in nodes:
        if nd['id'] not in depth_snp:
            depth_snp[nd['id']] = 0.0

    max_depth = max(depth_snp.values()) or 1.0

    # ── Count leaves (nodes with no children) in each subtree ────────────────
    def count_leaves(nid: int) -> int:
        ch = [c for c in children_of[nid] if c in all_ids]
        return sum(count_leaves(c) for c in ch) if ch else 1

    # ── Top-down DFS: allocate vertical band per subtree ─────────────────────
    y_pos: Dict[int, float] = {}

    def assign_y(nid: int, lo: float, hi: float) -> None:
        ch = [c for c in children_of[nid] if c in all_ids]
        if not ch:
            y_pos[nid] = (lo + hi) / 2.0
            return
        total  = count_leaves(nid) or 1
        cursor = lo
        for c in ch:
            share = (hi - lo) * count_leaves(c) / total
            assign_y(c, cursor, cursor + share)
            cursor += share
        # Parent y = mean of children y
        y_pos[nid] = sum(y_pos[c] for c in ch) / len(ch)

    assign_y(ancestor_id, padding, canvas_h - padding)

    # ── Write final coordinates onto nodes ───────────────────────────────────
    pixels_per_snp = (canvas_w - 2.0 * padding) / max_depth

    for nd in nodes:
        nid    = nd['id']
        nd['x'] = float(padding + depth_snp.get(nid, 0.0) * pixels_per_snp)
        nd['y'] = float(y_pos.get(nid, canvas_h / 2.0))

    # target_len for JS spring relaxer (px proportional to SNP distance)
    for e in edges:
        e['target_len'] = round(float(e.get('dist', 0)) * pixels_per_snp, 2)

    return nodes


def _repulse(
    positions: np.ndarray,          # (n, 2)  – modified in-place
    radii: np.ndarray,               # (n,)    – node radius for collision
    canvas_w: float,
    canvas_h: float,
    iterations: int = 80,
    padding: float  = 20.0,
) -> np.ndarray:

    """
    Simple O(n²) repulsion pass.  Pushes overlapping nodes apart and
    keeps them inside [padding, canvas-padding].  Returns positions.
    """

    n = len(positions)
    if n <= 1:
        return positions

    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx   = positions[i, 0] - positions[j, 0]
                dy   = positions[i, 1] - positions[j, 1]
                dist = math.hypot(dx, dy)
                min_d = radii[i] + radii[j] + 6.0   # 6 px gap
                if dist < min_d and dist > 1e-9:
                    overlap = (min_d - dist) / 2.0
                    ux, uy  = dx / dist, dy / dist
                    positions[i, 0] += ux * overlap
                    positions[i, 1] += uy * overlap
                    positions[j, 0] -= ux * overlap
                    positions[j, 1] -= uy * overlap
                    moved = True
                elif dist < 1e-9:               # exact overlap – jitter
                    positions[i, 0] += 1.5
                    positions[j, 0] -= 1.5
                    moved = True

        # clamp to canvas
        positions[:, 0] = np.clip(positions[:, 0], padding, canvas_w - padding)
        positions[:, 1] = np.clip(positions[:, 1], padding, canvas_h - padding)

        if not moved:
            break

    return positions


def _mds_layout(
    nodes: List[dict],
    edges: List[dict],
    canvas_w: float = 800.0,
    canvas_h: float = 700.0,
    padding: float  = 70.0,
) -> List[dict]:

    """
    Classical MDS on shortest-path Hamming distances between MJ nodes,
    followed by a repulsion pass to prevent overlap.
    """

    n = len(nodes)
    if n == 0:
        return nodes
    if n == 1:
        nodes[0]['x'] = canvas_w / 2
        nodes[0]['y'] = canvas_h / 2
        return nodes

    D = np.zeros((n, n))
    for e in edges:
        i, j, d = e['source'], e['target'], e['dist']
        D[i, j] = d
        D[j, i] = d

    INF = 1e9
    SP  = np.where(D > 0, D, INF)
    np.fill_diagonal(SP, 0.0)
    for k in range(n):
        SP = np.minimum(SP, SP[:, k:k+1] + SP[k:k+1, :])
    SP = np.where(SP >= INF, 0.0, SP)

    D2 = SP ** 2
    H  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * H @ D2 @ H
    B  = (B + B.T) / 2

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    pos = np.zeros((n, 2))
    for axis in range(2):
        ev = eigvals[idx[axis]]
        if ev > 0:
            pos[:, axis] = eigvecs[:, idx[axis]] * math.sqrt(ev)

    mn  = pos.min(axis=0)
    mx  = pos.max(axis=0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    pos = (pos - mn) / rng
    pos[:, 0] = pos[:, 0] * (canvas_w  - 2 * padding) + padding
    pos[:, 1] = pos[:, 1] * (canvas_h - 2 * padding) + padding

    # ── repulsion pass ────────────────────────────────────────────────────────
    radii = np.array([nd.get('size', 10) for nd in nodes], dtype=float)
    pos   = _repulse(pos, radii, canvas_w, canvas_h, padding=padding)

    for i, nd in enumerate(nodes):
        nd['x'] = float(pos[i, 0])
        nd['y'] = float(pos[i, 1])

    # ── compute px_per_snp from the actual MDS positions ─────────────────────
    # Median ratio of Euclidean distance / SNP distance across all edges;
    # avoids outliers from very short or very long branches.
    ratios = []
    for e in edges:
        snp = e.get('dist', 0)
        if snp <= 0:
            continue
        xi, yi = float(pos[e['source'], 0]), float(pos[e['source'], 1])
        xj, yj = float(pos[e['target'], 0]), float(pos[e['target'], 1])
        ratios.append(math.hypot(xj - xi, yj - yi) / snp)

    px_per_snp = float(np.median(ratios)) if ratios else 20.0
    px_per_snp = max(8.0, min(px_per_snp, 120.0))   # clamp to sensible range

    # Store the target pixel length on every edge so the JS spring relaxer
    # knows exactly how long each branch should be.
    for e in edges:
        e['target_len'] = round(e.get('dist', 0) * px_per_snp, 2)

    return nodes


# ─── Mode B: FR layout ────────────────────────────────────────────────────────
def _fr_layout(
    node_ids: List[int],
    edge_list: List[Tuple[int, int]],
    node_sizes: Optional[List[float]] = None,
    canvas_w: float = 0.0,
    canvas_h: float = 0.0,
    iterations: int = 400,
    seed: int       = 42,
) -> Tuple[Dict[int, Tuple[float, float]], float, float]:

    """
    Two-phase Fruchterman-Reingold layout for Mode B.

    Phase 1 — Global FR on super-nodes (one per connected component):
      • Repulsion uses node_r (not local_r) — balanced dynamics, large clusters
        don't over-repel and drift to the periphery.
      • Gravity is proportional to log(component size) — larger clusters are
        pulled more strongly toward the centre.

    Phase 1 post-process — sequential (Gauss-Seidel) geometric separation:
      • Processes pairs one at a time (largest-first) using local_r as the
        exclusion zone, guaranteeing zero overlap after Phase 2 expansion.
      • Sequential updates avoid the batch-accumulation issue where conflicting
        pair displacements cancel each other out.

    Phase 2 — Local component expansion:
      • Each multi-node component is placed in a golden-angle spiral around its
        super-node position with radius = local_r.

    Result: connected nodes cluster together, edges never cross foreign nodes,
    large clusters stay in the nebula centre, global distribution is preserved.
    """

    n = len(node_ids)
    if n == 0:
        return {}, max(canvas_w, 1400.0), max(canvas_h, 1000.0)

    radii  = np.array(node_sizes if node_sizes else [10.0] * n, dtype=float)
    avg_r  = float(radii.mean())
    max_r  = float(radii.max())
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # ── Auto-size canvas ──────────────────────────────────────────────────────
    if canvas_w <= 0 or canvas_h <= 0:
        area_needed = n * (avg_r * 11) ** 2
        side        = math.sqrt(area_needed) * 1.2
        side        = max(800.0, min(side, 8000.0))
        canvas_w    = side
        canvas_h    = side * 0.78

    if n == 1:
        return {node_ids[0]: (canvas_w / 2, canvas_h / 2)}, canvas_w, canvas_h

    pad    = max(60.0, max_r * 2)
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    # ── Build connected components via union-find ─────────────────────────────
    valid = [(u, v) for u, v in edge_list if u in id2idx and v in id2idx]

    uf: Dict[int, int] = {nid: nid for nid in node_ids}
    def _find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x
    for u, v in valid:
        a, b = _find(u), _find(v)
        if a != b: uf[a] = b

    comp_members: Dict[int, List[int]] = {}
    for nid in node_ids:
        r = _find(nid)
        if r not in comp_members:
            comp_members[r] = []
        comp_members[r].append(nid)

    # ── Per-component radii ───────────────────────────────────────────────────
    def _local_r(members: List[int]) -> float:
        m_radii = radii[np.array([id2idx[m] for m in members])]
        max_r_c = float(m_radii.max())
        nm      = len(members)
        if nm == 1:
            return max_r_c
        return max(max_r_c * 2.5, nm * (max_r_c + 4) / math.pi)

    super_ids     = list(comp_members.keys())
    ns            = len(super_ids)
    super_idx     = {sid: i for i, sid in enumerate(super_ids)}

    super_node_r  = np.array(
        [float(radii[np.array([id2idx[m] for m in comp_members[sid]])].max())
         for sid in super_ids], dtype=float)
    super_local_r = np.array(
        [_local_r(comp_members[sid]) for sid in super_ids], dtype=float)
    super_size    = np.array(
        [len(comp_members[sid]) for sid in super_ids], dtype=float)

    # ── Phase 1: FR on super-nodes ────────────────────────────────────────────
    rng  = np.random.default_rng(seed)
    cols = max(1, math.ceil(math.sqrt(ns)))
    rows = max(1, math.ceil(ns / cols))
    cw_g = (canvas_w - 2 * pad) / cols
    ch_g = (canvas_h - 2 * pad) / rows
    spos = np.zeros((ns, 2))
    for k_i in range(ns):
        col = k_i % cols
        row = k_i // cols
        x   = pad + cw_g * (col + 0.5) + rng.uniform(-cw_g * 0.35, cw_g * 0.35)
        y   = pad + ch_g * (row + 0.5) + rng.uniform(-ch_g * 0.35, ch_g * 0.35)
        spos[k_i, 0] = np.clip(x, pad, canvas_w - pad)
        spos[k_i, 1] = np.clip(y, pad, canvas_h - pad)

    area  = canvas_w * canvas_h
    k_fr  = math.sqrt(area / ns)
    r_cut = k_fr * 2.5
    temp  = min(canvas_w, canvas_h) * 0.10

    log_size      = np.log1p(super_size)
    log_size_norm = log_size / log_size.mean()

    for it in range(iterations):
        disp = np.zeros((ns, 2))

        # Repulsion — node_r keeps dynamics balanced
        tree  = cKDTree(spos)
        pairs = tree.query_pairs(r=r_cut, output_type='ndarray')
        if len(pairs):
            ia, ja = pairs[:, 0], pairs[:, 1]
            dxp    = spos[ia, 0] - spos[ja, 0]
            dyp    = spos[ia, 1] - spos[ja, 1]
            dp     = np.maximum(np.hypot(dxp, dyp), 0.01)
            ri_sum = super_node_r[ia] + super_node_r[ja]
            f      = (k_fr * k_fr + ri_sum * k_fr * 0.9) / dp
            ux     = dxp / dp;  uy = dyp / dp
            np.add.at(disp, (ia, 0),  ux * f)
            np.add.at(disp, (ia, 1),  uy * f)
            np.add.at(disp, (ja, 0), -ux * f)
            np.add.at(disp, (ja, 1), -uy * f)

        # Gravity — proportional to log(size), anchors larger clusters centrally
        base_grav = k_fr * max(0.004, 0.025 - ns * 0.000008)
        disp[:, 0] += (cx - spos[:, 0]) * base_grav * log_size_norm
        disp[:, 1] += (cy - spos[:, 1]) * base_grav * log_size_norm

        d_norm = np.maximum(np.hypot(disp[:, 0], disp[:, 1]), 0.01)
        scale  = np.minimum(d_norm, temp) / d_norm
        spos  += disp * scale[:, None]
        spos[:, 0] = np.clip(spos[:, 0], pad, canvas_w - pad)
        spos[:, 1] = np.clip(spos[:, 1], pad, canvas_h - pad)

        if not np.isfinite(spos).all():
            spos = np.nan_to_num(spos, nan=cx,
                                 posinf=canvas_w - pad, neginf=pad)

        temp *= 0.975 if it < int(iterations * 0.7) else 0.91

    # ── Phase 1 post-process: sequential (Gauss-Seidel) separation ───────────
    # Sort pairs by required gap descending so the largest conflicts are
    # resolved first; sequential updates propagate corrections immediately
    # instead of accumulating conflicting displacements in one batch step.
    max_search = float(super_local_r.max() * 2 + 20)
    for outer in range(400):
        tree  = cKDTree(spos)
        pairs = tree.query_pairs(r=max_search, output_type='ndarray')
        if not len(pairs): break

        # Compute required gaps for all pairs
        ia, ja   = pairs[:, 0], pairs[:, 1]
        gap_req  = super_local_r[ia] + super_local_r[ja] + 20.0
        dxp      = spos[ia, 0] - spos[ja, 0]
        dyp      = spos[ia, 1] - spos[ja, 1]
        dp_arr   = np.hypot(dxp, dyp)
        overlap  = gap_req - dp_arr
        in_conflict = overlap > 0
        if not in_conflict.any(): break

        # Sort by overlap descending — resolve largest conflicts first
        order = np.argsort(-overlap)

        changed = False
        for idx in order:
            if overlap[idx] <= 0: continue
            i, j = int(ia[idx]), int(ja[idx])
            dx = spos[i, 0] - spos[j, 0]
            dy = spos[i, 1] - spos[j, 1]
            dist = math.hypot(dx, dy)
            req  = super_local_r[i] + super_local_r[j] + 20.0
            if dist >= req: continue          # already resolved by earlier pair
            if dist < 1e-9:
                dx, dy, dist = 1.0, 0.0, 1.0
            ux, uy = dx / dist, dy / dist
            push   = (req - dist) / 2.0
            spos[i, 0] = np.clip(spos[i, 0] + ux * push, pad, canvas_w - pad)
            spos[i, 1] = np.clip(spos[i, 1] + uy * push, pad, canvas_h - pad)
            spos[j, 0] = np.clip(spos[j, 0] - ux * push, pad, canvas_w - pad)
            spos[j, 1] = np.clip(spos[j, 1] - uy * push, pad, canvas_h - pad)
            changed = True

        if not changed: break

    # ── Phase 2: expand multi-node components locally ─────────────────────────
    pos    = np.zeros((n, 2))
    golden = math.pi * (3.0 - math.sqrt(5.0))

    for sid, members in comp_members.items():
        si     = super_idx[sid]
        centre = spos[si].copy()
        m_idxs = [id2idx[m] for m in members]
        nm     = len(members)

        if nm == 1:
            pos[m_idxs[0]] = centre
        else:
            local_r = _local_r(members)
            for k_i, mi in enumerate(m_idxs):
                angle      = k_i * golden
                pos[mi, 0] = np.clip(
                    centre[0] + local_r * math.cos(angle), pad, canvas_w - pad)
                pos[mi, 1] = np.clip(
                    centre[1] + local_r * math.sin(angle), pad, canvas_h - pad)

    return (
        {nid: (float(pos[id2idx[nid], 0]), float(pos[id2idx[nid], 1]))
         for nid in node_ids},
        canvas_w,
        canvas_h,
    )


# ─── Main network builder ─────────────────────────────────────────────────────
MJ_MAX = 200   # keep in sync with the rest of the module

def _build_network(
    dist_matrix: "np.ndarray",
    sample_names: List[str],
    genomic_threshold: int,
    collapse_threshold: int,
    output_file: str,
    snp_matrix_T: Optional["np.ndarray"] = None,
) -> None:
 
    from collections import defaultdict as _dd
    import numpy as _np
 
    n = len(sample_names)
 
    # ── cluster assignment ────────────────────────────────────────────────────
    def _clusters(dist, ns, thr):
        parent = list(range(ns))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(ns):
            for j in range(i + 1, ns):
                if dist[i, j] <= thr:
                    a, b = find(i), find(j)
                    if a != b: parent[a] = b
        g = _dd(list)
        for i in range(ns):
            g[find(i)].append(i)
        return dict(g)
 
    raw  = _clusters(dist_matrix, n, collapse_threshold)
    grps = sorted(raw.values(), key=len, reverse=True)
    cid_members: Dict[int, List[int]] = {cid: m for cid, m in enumerate(grps)}
    n_clusters   = sum(1 for m in cid_members.values() if len(m) >= 2)
    n_singletons = sum(1 for m in cid_members.values() if len(m) == 1)
    mode_a       = (len(cid_members) == 1)
 
    # =========================================================================
    # MODE A
    # =========================================================================
    if mode_a:
        ancestor_node_id = None
 
        if snp_matrix_T is not None and n <= MJ_MAX:
            # ─── 1. Binary conversion: 0.5 → 0 ──────────────────────────────
            hap_clean = (_np.array(snp_matrix_T) > 0.5).astype(_np.int8)
 
            # ─── 2. Identify positions shared by ALL samples (N == n) ─────────
            #        In hap_clean: position j is "universal" iff every sample
            #        has value 1 there (equivalent to N == total in the DataFrame).
            all_have        = _np.all(hap_clean > 0, axis=0)   # (n_positions,)
            universal_idx   = _np.where(all_have)[0]
            informative_idx = _np.where(~all_have)[0]
 
            # Keep exactly 1 representative universal position (to guarantee
            # the all-zero ancestor vector is always at distance ≥ 1 from every
            # sample) plus all discriminating positions.
            if len(universal_idx) > 0:
                keep_idx = _np.concatenate([[universal_idx[0]], informative_idx])
            elif len(informative_idx) > 0:
                keep_idx = informative_idx
            else:
                keep_idx = _np.arange(hap_clean.shape[1])
 
            hap = hap_clean[:, keep_idx]          # (n_samples, n_kept_positions)
 
            # ─── 3. Ancestor haplotype: all zeros ────────────────────────────
            #        Pre-mutation state; every SNP in the matrix is derived
            #        (occurred after) this node.
            ancestor_hap = _np.zeros(hap.shape[1], dtype=_np.int8)
 
            # ─── 4. Run MJ with ancestor prepended as a virtual sample ────────
            all_names_mj = ['__ANCESTOR__'] + list(sample_names)
            all_haps_mj  = _np.vstack([ancestor_hap[_np.newaxis, :], hap])
            mj_nodes, mj_edges = _mj(all_names_mj, all_haps_mj)
 
            # ─── 5. Style the ancestor node ───────────────────────────────────
            for nd in mj_nodes:
                if '__ANCESTOR__' in nd.get('members', []):
                    co = [m for m in nd['members'] if m != '__ANCESTOR__']
                    tooltip = 'Common ancestor (SNPs shared by all the samples)'
                    if co:
                        tooltip += '\n\nCo-located samples:\n' + '\n'.join(co)
                    nd.update({
                        'is_ancestor': True,
                        'is_mv'      : False,
                        'color'      : '#FFD700',
                        'stroke'     : '#FF8C00',
                        'size'       : 16,
                        'label'      : 'Ancestor',
                        'title'      : tooltip,
                        'members'    : ['Common ancestor'] + co,
                    })
                    ancestor_node_id = nd['id']
                    break
 
        else:
            # Fallback: simple MST when snp_matrix_T is absent or n > MJ_MAX.
            # Ancestor is node 0; samples are nodes 1..n.
            mj_nodes = [{
                'id'         : 0,
                'label'      : 'Ancestor',
                'title'      : 'Common ancestor (SNPs shared by all the samples)',
                'members'    : ['Common ancestor'],
                'is_mv'      : False,
                'is_ancestor': True,
                'singleton'  : False,
                'color'      : '#FFD700',
                'stroke'     : '#FF8C00',
                'size'       : 16,
            }] + [
                {
                    'id'         : i + 1,
                    'label'      : nm,
                    'title'      : nm,
                    'members'    : [nm],
                    'is_mv'      : False,
                    'is_ancestor': False,
                    'singleton'  : False,
                    'color'      : '#1976D2',
                    'stroke'     : '#0d47a1',
                    'size'       : 12,
                }
                for i, nm in enumerate(sample_names)
            ]
 
            # Build MST (node i+1  ↔  dist_matrix row/col i, 0-indexed)
            in_tree   = {1}
            remaining = set(range(2, n + 1))
            mst_raw   = []
            while remaining:
                best = None
                for u in in_tree:
                    for v in remaining:
                        d = int(dist_matrix[u - 1, v - 1])
                        if best is None or d < best[2]:
                            best = (u, v, d)
                if best:
                    mst_raw.append(best)
                    in_tree.add(best[1])
                    remaining.discard(best[1])
 
            max_d    = max((d for _, _, d in mst_raw), default=1) or 1
            mj_edges = [
                {'source': a, 'target': b, 'dist': d,
                 'label': str(d), 'width': max(1.2, 1.5 + 4.5 * d / max_d)}
                for a, b, d in mst_raw
            ]
 
            # Connect ancestor to the most central sample (min summed distance)
            if n > 0:
                root_samp = int(_np.argmin(dist_matrix.sum(axis=1)))
                root_node = root_samp + 1
                d_stem    = int(dist_matrix[root_samp].min())
                mj_edges.append({
                    'source': 0, 'target': root_node,
                    'dist'  : d_stem, 'label': str(d_stem), 'width': 1.5,
                })
                ancestor_node_id = 0
 
        # ── Layout: rooted tree (left → right) when ancestor is present ───────
        if ancestor_node_id is not None:
            mj_nodes = _tree_layout_rooted(mj_nodes, mj_edges, ancestor_node_id)
        else:
            mj_nodes = _mds_layout(mj_nodes, mj_edges)
 
        mode       = 'A'
        nodes_json = json.dumps(mj_nodes)
        edges_json = json.dumps(mj_edges)
        title_str  = f'Median Joining - {n} samples'
        fr_cw_js   = 1400.0
        fr_ch_js   = 1000.0
 
    # =========================================================================
    # MODE B
    # =========================================================================
    else:
        ov_nodes: List[dict] = []
        for cid, members in cid_members.items():
            sz    = len(members)
            names = [sample_names[i] for i in members]
            single = sz == 1
            ov_nodes.append({
                'id'       : cid,
                'label'    : names[0] if single else f'C{cid+1} ({sz})',
                'title'    : '\n'.join(names[:80]) + ('\n...' if sz > 80 else ''),
                'is_mv'    : False,
                'singleton': single,
                'color'    : '#9e9e9e' if single else '#e53935',
                'stroke'   : '#757575' if single else '#b71c1c',
                'size'     : 7 if single else min(10 + sz * 2, 55),
                'n'        : sz,
                'members'  : names,
            })
 
        # edges: all pairs within genomic_threshold
        nc = len(cid_members)
        ov_edges_data: List[Tuple[int, int, int]] = []
        for i in range(nc):
            for j in range(i + 1, nc):
                ia = cid_members[i]; ib = cid_members[j]
                md = int(_np.min(dist_matrix[_np.ix_(ia, ib)]))
                if md <= genomic_threshold:
                    ov_edges_data.append((i, j, md))
 
        # FR layout (auto-sized canvas)
        node_ids_fr   = [nd['id'] for nd in ov_nodes]
        node_sizes_fr = [nd['size'] for nd in ov_nodes]
        edge_list_fr  = [(a, b) for a, b, _ in ov_edges_data]
        iters = min(600, max(300, len(node_ids_fr) // 2))
        pos_fr, fr_canvas_w, fr_canvas_h = _fr_layout(
            node_ids_fr, edge_list_fr, node_sizes=node_sizes_fr,
            iterations=iters,
        )
 
        for nd in ov_nodes:
            x, y   = pos_fr.get(nd['id'], (700.0, 500.0))
            nd['x'] = x; nd['y'] = y
 
        max_gt = max(genomic_threshold, 1)
        ov_edges = [
            {'source': a, 'target': b, 'dist': d,
             'label' : str(d),
             'width' : max(0.8, 3.0 * (1.0 - d / max_gt))}
            for a, b, d in ov_edges_data
        ]
 
        mode = 'B'
        nodes_json = json.dumps(ov_nodes)
        edges_json = json.dumps(ov_edges)
        title_str  = f'Genomic Network - {n} samples'
        fr_cw_js   = round(fr_canvas_w, 1)
        fr_ch_js   = round(fr_canvas_h, 1)
 
    # =========================================================================
    # HTML
    # =========================================================================
    legend_a = (
        '<div class="lbl">Legend</div>'
        '<div style="font-size:10px;line-height:2.1">'
        '<span class="star-dot"></span>Ancestor&nbsp;&nbsp;'
        '<span class="dot" style="background:#1976D2"></span>Sample&nbsp;&nbsp;'
        '<span class="mv-dot"></span>Median Vector'
        '</div>'
    )
    legend_b = (
        '<div class="lbl">Legend</div>'
        '<div style="font-size:10px;line-height:2.1">'
        '<span class="dot" style="background:#e53935"></span>Cluster&nbsp;&nbsp;'
        '<span class="dot" style="background:#9e9e9e"></span>Singleton'
        '</div>'
    )
    legend_html = legend_a if mode == 'A' else legend_b
    mode_badge  = 'Mode A - Median Joining' if mode == 'A' else 'Mode B - Overview'
    HL_COLOR    = '#eeee0c'
 
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Genomic Network</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #fff; font-family: 'Courier New', monospace;
        color: #37474f; height: 100vh; overflow: hidden; }}
 
/* toolbar */
#tb {{ position: fixed; top: 0; left: 0; right: 0; height: 46px;
      background: #f5f5f5; border-bottom: 1px solid #ddd;
      display: flex; align-items: center; gap: 8px;
      padding: 0 14px; z-index: 10; flex-wrap: nowrap; }}
#tb h1 {{ font-size: 11px; color: #1565C0; letter-spacing: 2px;
          text-transform: uppercase; font-weight: bold; white-space: nowrap; }}
.sep {{ width: 1px; height: 20px; background: #ddd; flex-shrink: 0; }}
button {{ background: #fff; color: #37474f; border: 1px solid #bdbdbd;
          border-radius: 4px; padding: 4px 10px;
          font-family: 'Courier New', monospace; font-size: 11px;
          cursor: pointer; white-space: nowrap; transition: all .12s; }}
button:hover  {{ background: #e8f5e9; border-color: #2e7d32; color: #2e7d32; }}
button.on     {{ background: #2e7d32; color: #fff; border-color: #2e7d32; }}
#srch {{ border: 1px solid #bdbdbd; border-radius: 4px; padding: 4px 8px;
         font-family: 'Courier New', monospace; font-size: 11px;
         width: 150px; outline: none; }}
#srch:focus {{ border-color: #2e7d32; }}
#flex {{ flex: 1; }}
.badge {{ font-size: 10px; background: #e8f5e9; color: #2e7d32;
          border: 1px solid #a5d6a7; border-radius: 3px;
          padding: 2px 7px; white-space: nowrap; }}
 
/* layout */
#canvas {{ position: fixed; top: 46px; left: 0; right: 280px; bottom: 0; }}
#panel  {{ position: fixed; top: 46px; right: 0; width: 280px; bottom: 0;
           background: #fafafa; border-left: 1px solid #e0e0e0;
           display: flex; flex-direction: column; overflow: hidden; }}
#ph {{ padding: 12px 14px 8px; border-bottom: 1px solid #e0e0e0; flex-shrink: 0; }}
#ph h2 {{ font-size: 11px; color: #1565C0; text-transform: uppercase; letter-spacing: 1px; }}
#pb {{ flex: 1; overflow-y: auto; padding: 10px 14px; font-size: 11px; line-height: 1.9; }}
.sr  {{ display: flex; justify-content: space-between; color: #78909c; }}
.sv  {{ color: #263238; font-weight: bold; }}
.lbl {{ font-size: 10px; color: #b0bec5; text-transform: uppercase;
        letter-spacing: 1px; margin: 8px 0 3px; }}
.hlist {{ font-size: 10px; max-height: 200px; overflow-y: auto;
          background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
          padding: 5px 7px; line-height: 1.85; }}
.hi {{ color: #78909c; }}
.hi.found {{ color: #2e7d32; font-weight: bold; }}
 
svg {{ width: 100%; height: 100%; }}
.lnk {{ fill: none; stroke: #1976D2; stroke-width: 1.5; }}
.nd  {{ cursor: pointer; }}
.nd circle, .nd polygon {{ transition: stroke-width .15s; }}
.nd:hover circle, .nd:hover polygon {{ stroke-width: 3.5 !important; }}
.nd.sel circle, .nd.sel polygon {{
  stroke: {HL_COLOR} !important;
  stroke-width: 3 !important;
}}
/* Node labels: appear ABOVE the node to avoid overlap with edge labels */
.nlbl {{ font-family: 'Courier New', monospace; pointer-events: none;
         text-anchor: middle; dominant-baseline: auto; fill: #546e7a; }}
/* Edge labels: white background pill for readability */
.elbl-bg {{ fill: rgba(255,255,255,0.85); stroke: none; }}
.elbl {{ font-family: 'Courier New', monospace; fill: #546e7a;
         font-size: 9px; pointer-events: none; text-anchor: middle;
         dominant-baseline: central; }}
 
#tip {{ position: fixed; background: #fff; border: 1px solid #1976D2;
        border-radius: 5px; padding: 7px 11px; font-size: 11px; color: #263238;
        pointer-events: none; opacity: 0; max-width: 240px; z-index: 200;
        white-space: pre-wrap; box-shadow: 0 2px 8px rgba(0,0,0,.10);
        transition: opacity .1s; }}
 
#lasso {{ fill: rgba(46,125,50,.07); stroke: #2e7d32; stroke-width: 1.4;
          stroke-dasharray: 5 3; pointer-events: none; display: none; }}
 
#selbar {{ position: fixed; bottom: 12px; left: 12px; background: #f5f5f5;
           border: 1px solid #ddd; border-radius: 5px; padding: 7px 12px;
           font-size: 11px; color: #78909c; display: none;
           gap: 8px; align-items: center; }}
 
.dot    {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
           margin-right: 4px; vertical-align: middle; }}
.mv-dot {{ display: inline-block; width: 9px; height: 9px;
           clip-path: polygon(50% 0%,100% 50%,50% 100%,0% 50%);
           background: #b0bec5; margin-right: 4px; vertical-align: middle; }}
/* ★ Ancestor legend icon */
.star-dot {{ display: inline-block; width: 11px; height: 11px;
             clip-path: polygon(50% 0%,62% 33%,98% 35%,70% 57%,79% 91%,50% 71%,21% 91%,30% 57%,2% 35%,38% 33%);
             background: #FFD700; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
 
<div id="tb">
  <h1>Genomic Network</h1>
  <span class="badge">{mode_badge}</span>
  <div class="sep"></div>
  <input id="srch" type="text" placeholder="Search sample..." oninput="doSearch(this.value)">
  <div class="sep"></div>
  <button id="lbtn" onclick="toggleLasso()">Select</button>
  <button id="dlbtn" style="display:none" onclick="dlSel()">Download selection</button>
  <div id="flex"></div>
  <button onclick="dlSVG()">Download SVG</button>
</div>
 
<div id="canvas">
  <svg id="net">
    <g id="zg">
      <g id="lg"></g>
      <g id="eg"></g>
      <g id="ng"></g>
    </svg>
    <rect id="lasso"></rect>
  </svg>
</div>
 
<div id="panel">
  <div id="ph"><h2 id="ptitle">{title_str}</h2></div>
  <div id="pb">
    <div class="sr"><span>Total samples</span><span class="sv">{n}</span></div>
    <div class="sr"><span>Clusters (2+ samples)</span><span class="sv">{n_clusters}</span></div>
    <div class="sr"><span>Singletons</span><span class="sv">{n_singletons}</span></div>
    <div class="sr"><span>Collapse threshold</span><span class="sv">{collapse_threshold} SNPs</span></div>
    <div class="sr"><span>Genomic threshold</span><span class="sv">{genomic_threshold} SNPs</span></div>
    {legend_html}
    <div id="nodeinfo" style="display:none">
      <div class="lbl" id="nodeinfo-lbl">Node members</div>
      <div class="hlist" id="nodeinfo-list"></div>
    </div>
    <div id="selinfo" style="display:none">
      <div class="lbl">Selected samples</div>
      <div class="hlist" id="sellist"></div>
    </div>
  </div>
</div>
 
<div id="tip"></div>
<div id="selbar">
  Selected: <b id="selcount">0</b>
  <button onclick="dlSel()" style="padding:3px 9px">Download TXT</button>
</div>
 
<script>
var NODES = {nodes_json};
var EDGES = {edges_json};
var MODE  = '{mode}';
var HL    = '{HL_COLOR}';
var FR_CW = {fr_cw_js};
var FR_CH = {fr_ch_js};
 
var tip = document.getElementById('tip');
var svg = d3.select('#net');
var zg  = d3.select('#zg');
var lg  = d3.select('#lg');
var eg  = d3.select('#eg');
var ng  = d3.select('#ng');
 
var lassoOn = false, lassoP0 = null;
var sel = new Set();
 
/* ── Zoom / pan ── */
var zoom = d3.zoom().scaleExtent([0.005, 60])
  .on('zoom', function(e) {{ zg.attr('transform', e.transform); }});
svg.call(zoom);
 
/* ── Lasso ── */
function toggleLasso() {{
  lassoOn = !lassoOn;
  d3.select('#lbtn').classed('on', lassoOn);
  if (lassoOn) {{ svg.on('.zoom', null); }}
  else {{
    svg.call(zoom);
    d3.select('#lasso').style('display','none');
    lassoP0 = null;
  }}
}}
 
svg.on('mousedown', function(e) {{
  if (!lassoOn) return;
  e.preventDefault();
  lassoP0 = d3.pointer(e, svg.node());
  sel.clear();
  ng.selectAll('.nd').classed('sel', false);
  document.getElementById('selbar').style.display  = 'none';
  document.getElementById('dlbtn').style.display   = 'none';
  document.getElementById('selinfo').style.display = 'none';
}});
 
svg.on('mousemove', function(e) {{
  if (!lassoOn || !lassoP0) return;
  e.preventDefault();
  var p = d3.pointer(e, svg.node());
  d3.select('#lasso').style('display', null)
    .attr('x',      Math.min(lassoP0[0], p[0]))
    .attr('y',      Math.min(lassoP0[1], p[1]))
    .attr('width',  Math.abs(p[0] - lassoP0[0]))
    .attr('height', Math.abs(p[1] - lassoP0[1]));
}});
 
svg.on('mouseup', function(e) {{
  if (!lassoOn || !lassoP0) return;
  var p  = d3.pointer(e, svg.node());
  var x0 = Math.min(lassoP0[0], p[0]), y0 = Math.min(lassoP0[1], p[1]);
  var x1 = Math.max(lassoP0[0], p[0]), y1 = Math.max(lassoP0[1], p[1]);
  lassoP0 = null;
  d3.select('#lasso').style('display', 'none');
  var t = d3.zoomTransform(svg.node());
  ng.selectAll('.nd').each(function(d) {{
    if (d.x == null) return;
    var sx = t.applyX(d.x), sy = t.applyY(d.y);
    if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) {{
      (d.members || [d.label]).forEach(function(nm) {{
        if (nm && nm.trim()) sel.add(nm.trim());
      }});
      d3.select(this).classed('sel', true);
    }}
  }});
  if (sel.size > 0) {{
    document.getElementById('selcount').textContent  = sel.size;
    document.getElementById('selbar').style.display  = 'flex';
    document.getElementById('dlbtn').style.display   = 'inline-block';
    var arr = Array.from(sel);
    document.getElementById('sellist').innerHTML =
      arr.slice(0, 150).map(function(s) {{
        return '<span class="hi">' + s + '</span>';
      }}).join('<br>') +
      (arr.length > 150
        ? '<br><span style="color:#b0bec5">...and ' + (arr.length-150) + ' more</span>'
        : '');
    document.getElementById('selinfo').style.display = 'block';
  }}
}});
 
function dlSel() {{
  var b = new Blob([Array.from(sel).join('\\n')], {{type:'text/plain'}});
  var u = URL.createObjectURL(b);
  var a = document.createElement('a'); a.href = u; a.download = 'selected.txt'; a.click();
  URL.revokeObjectURL(u);
}}
 
/* ── Search ── */
function doSearch(q) {{
  q = q.trim().toLowerCase();
  ng.selectAll('.nd circle, .nd polygon')
    .attr('stroke', function(d) {{
      if (!q) return d.stroke;
      return ((d.label||'').toLowerCase().includes(q)||(d.title||'').toLowerCase().includes(q))
        ? HL : d.stroke;
    }})
    .attr('stroke-width', function(d) {{
      if (!q) return 1.8;
      return ((d.label||'').toLowerCase().includes(q)||(d.title||'').toLowerCase().includes(q))
        ? 3.5 : 1.8;
    }});
}}
 
/* ── Node click ── */
function showNodeInfo(d) {{
  if (d.is_mv) return;
  var members = d.members || [];
  if (!members.length) return;
  var lbl = d.is_ancestor
    ? 'Common Ancestor'
    : (members.length === 1 ? 'Sample' : 'Cluster members (' + members.length + ')');
  document.getElementById('nodeinfo-lbl').textContent = lbl;
  document.getElementById('nodeinfo-list').innerHTML =
    members.slice(0,200).map(function(s) {{
      return '<span class="hi">' + s + '</span>';
    }}).join('<br>') +
    (members.length > 200
      ? '<br><span style="color:#b0bec5">...and ' + (members.length-200) + ' more</span>'
      : '');
  document.getElementById('nodeinfo').style.display = 'block';
}}
 
/* ── Diamond helper (Median Vectors) ── */
function dpts(r) {{
  return '0,' + (-r) + ' ' + r + ',0 0,' + r + ' ' + (-r) + ',0';
}}
 
/* ── Star helper (Ancestor node) ──
   5-pointed star: outer radius r, inner radius 0.42*r               */
function starPts(r) {{
  var pts   = [];
  var inner = r * 0.42;
  for (var i = 0; i < 10; i++) {{
    var angle = (i * Math.PI / 5) - Math.PI / 2;
    var rad   = (i % 2 === 0) ? r : inner;
    pts.push(Math.cos(angle) * rad + ',' + Math.sin(angle) * rad);
  }}
  return pts.join(' ');
}}
 
/* ── Draw nodes (Mode B — with drag, no spring) ── */
function drawNodes(nodes, edges, idMap, edgeSel, edgeLabelSel) {{
  var node = ng.selectAll('g').data(nodes, function(d) {{ return d.id; }})
    .join('g').attr('class', 'nd');
 
  node.each(function(d) {{
    var g = d3.select(this);
    var r = d.size || 10;
    if (d.is_ancestor) {{
      g.append('polygon')
        .attr('points', starPts(r))
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 2.5);
    }} else if (d.is_mv) {{
      g.append('polygon')
        .attr('points', dpts(r))
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.5);
    }} else {{
      g.append('circle')
        .attr('r', r)
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.8);
    }}
  }});
 
  node.append('text').attr('class', 'nlbl')
    .attr('dy', function(d) {{ return -((d.size || 10) + 4); }})
    .attr('font-size', 9)
    .text(function(d) {{
      if (d.is_mv || d.singleton) return '';
      return d.label;
    }});
 
  var dragBehaviour = d3.drag()
    .on('start', function(e, d) {{
      svg.on('.zoom', null);
      tip.style.opacity = '0';
      d3.select(this).raise();
    }})
    .on('drag', function(e, d) {{
      var t  = d3.zoomTransform(svg.node());
      d.x += e.dx / t.k;
      d.y += e.dy / t.k;
      d3.select(this).attr('transform', 'translate(' + d.x + ',' + d.y + ')');
      if (edgeSel) {{
        edgeSel
          .attr('x1', function(ed) {{ return (idMap[ed.source]||{{}}).x||0; }})
          .attr('y1', function(ed) {{ return (idMap[ed.source]||{{}}).y||0; }})
          .attr('x2', function(ed) {{ return (idMap[ed.target]||{{}}).x||0; }})
          .attr('y2', function(ed) {{ return (idMap[ed.target]||{{}}).y||0; }});
      }}
      if (edgeLabelSel) {{
        var LOFF = 10;
        edgeLabelSel.selectAll('rect')
          .attr('x', function(ed) {{
            var s = idMap[ed.source]||{{}}, t2 = idMap[ed.target]||{{}};
            return edgeLabelPos(s.x||0,s.y||0,t2.x||0,t2.y||0,LOFF).x - 11;
          }})
          .attr('y', function(ed) {{
            var s = idMap[ed.source]||{{}}, t2 = idMap[ed.target]||{{}};
            return edgeLabelPos(s.x||0,s.y||0,t2.x||0,t2.y||0,LOFF).y - 6.5;
          }});
        edgeLabelSel.selectAll('text')
          .attr('x', function(ed) {{
            var s = idMap[ed.source]||{{}}, t2 = idMap[ed.target]||{{}};
            return edgeLabelPos(s.x||0,s.y||0,t2.x||0,t2.y||0,LOFF).x;
          }})
          .attr('y', function(ed) {{
            var s = idMap[ed.source]||{{}}, t2 = idMap[ed.target]||{{}};
            return edgeLabelPos(s.x||0,s.y||0,t2.x||0,t2.y||0,LOFF).y;
          }});
      }}
    }})
    .on('end', function() {{
      if (!lassoOn) svg.call(zoom);
    }});
 
  node.call(dragBehaviour);
 
  node.on('mouseover', function(e, d) {{
      tip.style.opacity = '1';
      var members = d.members || [];
      tip.textContent = members.length
        ? members.slice(0,8).join('\\n') + (members.length > 8 ? '\\n...+' + (members.length-8) + ' more' : '')
        : (d.title || d.label || '');
    }})
    .on('mousemove', function(e) {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 10) + 'px';
    }})
    .on('mouseout',  function() {{ tip.style.opacity = '0'; }})
    .on('click',     function(e, d) {{
      if (!e.defaultPrevented) showNodeInfo(d);
    }});
 
  return node;
}}
 
/* ── Edge label at midpoint with perpendicular offset ── */
function edgeLabelPos(x1, y1, x2, y2, offset) {{
  var mx = (x1 + x2) / 2;
  var my = (y1 + y2) / 2;
  var dx = x2 - x1, dy = y2 - y1;
  var len = Math.sqrt(dx*dx + dy*dy) || 1;
  var px = -dy / len, py = dx / len;
  return {{ x: mx + px * offset, y: my + py * offset }};
}}
 
/* ── Spring relaxation (Mode A) ─────────────────────────────────────────────
   After a node is dragged to position P, all OTHER nodes (except the
   ancestor, which stays pinned) spring-relax to restore proportional
   branch lengths.
──────────────────────────────────────────────────────────────────────────── */
function springRelax(nodes, edges, idMap, pinnedId) {{
  var ITERS   = 120;
  var ALPHA   = 0.25;
  var REPULSE = 800;
 
  for (var iter = 0; iter < ITERS; iter++) {{
    var disp = {{}};
    nodes.forEach(function(d) {{ disp[d.id] = {{x:0, y:0}}; }});
 
    /* Spring forces */
    edges.forEach(function(e) {{
      var s  = idMap[e.source], t = idMap[e.target];
      if (!s || !t) return;
      var dx   = t.x - s.x, dy = t.y - s.y;
      var dist = Math.sqrt(dx*dx + dy*dy) || 0.01;
      var tgt  = e.target_len || dist;
      var f    = (dist - tgt) / dist;
      var fx   = f * dx, fy = f * dy;
      if (s.id !== pinnedId) {{ disp[s.id].x += fx; disp[s.id].y += fy; }}
      if (t.id !== pinnedId) {{ disp[t.id].x -= fx; disp[t.id].y -= fy; }}
    }});
 
    /* Repulsion forces */
    for (var i = 0; i < nodes.length; i++) {{
      for (var j = i+1; j < nodes.length; j++) {{
        var a = nodes[i], b = nodes[j];
        var dx = b.x - a.x, dy = b.y - a.y;
        var d2 = dx*dx + dy*dy || 0.01;
        var f  = REPULSE / d2;
        var fx = f * dx / Math.sqrt(d2), fy = f * dy / Math.sqrt(d2);
        if (a.id !== pinnedId) {{ disp[a.id].x -= fx; disp[a.id].y -= fy; }}
        if (b.id !== pinnedId) {{ disp[b.id].x += fx; disp[b.id].y += fy; }}
      }}
    }}
 
    /* Apply — skip ancestor (always pinned) and the dragged node */
    nodes.forEach(function(d) {{
      if (d.id === pinnedId) return;
      if (d.is_ancestor)     return;   /* ancestor stays where it was placed */
      d.x += disp[d.id].x * ALPHA;
      d.y += disp[d.id].y * ALPHA;
    }});
  }}
}}
 
/* ── Redraw helper: sync all DOM elements to current node positions ── */
function redrawModeA(nodes, edges, idMap, nodeSel, edgeSel, elg) {{
  var LOFF = 10;
  nodeSel.attr('transform', function(d) {{ return 'translate('+d.x+','+d.y+')'; }});
  edgeSel
    .attr('x1', function(e) {{ return (idMap[e.source]||{{}}).x||0; }})
    .attr('y1', function(e) {{ return (idMap[e.source]||{{}}).y||0; }})
    .attr('x2', function(e) {{ return (idMap[e.target]||{{}}).x||0; }})
    .attr('y2', function(e) {{ return (idMap[e.target]||{{}}).y||0; }});
  elg.selectAll('rect')
    .attr('x', function(e) {{
      var s=idMap[e.source]||{{}}, t=idMap[e.target]||{{}};
      return edgeLabelPos(s.x||0,s.y||0,t.x||0,t.y||0,LOFF).x - 11;
    }})
    .attr('y', function(e) {{
      var s=idMap[e.source]||{{}}, t=idMap[e.target]||{{}};
      return edgeLabelPos(s.x||0,s.y||0,t.x||0,t.y||0,LOFF).y - 6.5;
    }});
  elg.selectAll('text')
    .attr('x', function(e) {{
      var s=idMap[e.source]||{{}}, t=idMap[e.target]||{{}};
      return edgeLabelPos(s.x||0,s.y||0,t.x||0,t.y||0,LOFF).x;
    }})
    .attr('y', function(e) {{
      var s=idMap[e.source]||{{}}, t=idMap[e.target]||{{}};
      return edgeLabelPos(s.x||0,s.y||0,t.x||0,t.y||0,LOFF).y;
    }});
}}
 
/* ── Mode A ── */
function renderModeA(nodes, edges) {{
  var W = document.getElementById('canvas').offsetWidth;
  var H = document.getElementById('canvas').offsetHeight;
  var CW = 900, CH = 700;
  var sc   = Math.min(W / CW, H / CH) * 0.88;
  var offX = (W - CW * sc) / 2;
  var offY = (H - CH * sc) / 2;
  nodes.forEach(function(d) {{
    d.x = (d.x || 0) * sc + offX;
    d.y = (d.y || 0) * sc + offY;
  }});
  /* Scale target_len from Python layout space to screen space */
  edges.forEach(function(e) {{
    e.target_len = (e.target_len || 0) * sc;
  }});
 
  var idMap = {{}};
  nodes.forEach(function(d) {{ idMap[d.id] = d; }});
 
  /* Draw edges */
  lg.selectAll('line').data(edges).join('line')
    .attr('class', 'lnk')
    .attr('stroke', '#1976D2')
    .attr('stroke-width', function(d) {{ return d.width || 1.5; }})
    .attr('x1', function(d) {{ return (idMap[d.source]||{{}}).x||0; }})
    .attr('y1', function(d) {{ return (idMap[d.source]||{{}}).y||0; }})
    .attr('x2', function(d) {{ return (idMap[d.target]||{{}}).x||0; }})
    .attr('y2', function(d) {{ return (idMap[d.target]||{{}}).y||0; }});
 
  /* Edge labels with background rect + perpendicular offset */
  var LOFF = 10;
  var elg = eg.selectAll('g').data(edges).join('g');
  elg.append('rect').attr('class', 'elbl-bg')
    .attr('rx', 2).attr('ry', 2)
    .attr('width', 22).attr('height', 13)
    .attr('x', function(d) {{
      var s = idMap[d.source]||{{}}, t = idMap[d.target]||{{}};
      return edgeLabelPos(s.x||0, s.y||0, t.x||0, t.y||0, LOFF).x - 11;
    }})
    .attr('y', function(d) {{
      var s = idMap[d.source]||{{}}, t = idMap[d.target]||{{}};
      return edgeLabelPos(s.x||0, s.y||0, t.x||0, t.y||0, LOFF).y - 6.5;
    }});
  elg.append('text').attr('class', 'elbl')
    .attr('x', function(d) {{
      var s = idMap[d.source]||{{}}, t = idMap[d.target]||{{}};
      return edgeLabelPos(s.x||0, s.y||0, t.x||0, t.y||0, LOFF).x;
    }})
    .attr('y', function(d) {{
      var s = idMap[d.source]||{{}}, t = idMap[d.target]||{{}};
      return edgeLabelPos(s.x||0, s.y||0, t.x||0, t.y||0, LOFF).y;
    }})
    .text(function(d) {{ return d.label || ''; }});
 
  var edgeSel = lg.selectAll('line');
 
  /* ── Draw nodes ── */
  var node = ng.selectAll('g').data(nodes, function(d) {{ return d.id; }})
    .join('g').attr('class', 'nd');
 
  node.each(function(d) {{
    var g = d3.select(this);
    var r = d.size || 10;
    if (d.is_ancestor) {{
      /* ★ 5-pointed gold star */
      g.append('polygon')
        .attr('points', starPts(r))
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 2.5);
    }} else if (d.is_mv) {{
      /* ◆ Diamond for Median Vectors */
      g.append('polygon')
        .attr('points', dpts(r))
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.5);
    }} else {{
      /* ● Circle for real samples */
      g.append('circle')
        .attr('r', r)
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.8);
    }}
  }});
 
  node.append('text').attr('class', 'nlbl')
    .attr('dy', function(d) {{ return -((d.size || 10) + 4); }})
    .attr('font-size', 9)
    .text(function(d) {{
      if (d.is_mv || d.singleton) return '';
      return d.label;
    }});
 
  var dragA = d3.drag()
    .on('start', function(e, d) {{
      svg.on('.zoom', null);
      tip.style.opacity = '0';
      d3.select(this).raise();
    }})
    .on('drag', function(e, d) {{
      var t = d3.zoomTransform(svg.node());
      d.x += e.dx / t.k;
      d.y += e.dy / t.k;
      springRelax(nodes, edges, idMap, d.id);
      redrawModeA(nodes, edges, idMap, node, edgeSel, elg);
    }})
    .on('end', function() {{
      if (!lassoOn) svg.call(zoom);
    }});
 
  node.call(dragA);
 
  node.on('mouseover', function(e, d) {{
      tip.style.opacity = '1';
      var members = d.members || [];
      tip.textContent = members.length
        ? members.slice(0,8).join('\\n') + (members.length > 8 ? '\\n...+' + (members.length-8) + ' more' : '')
        : (d.title || d.label || '');
    }})
    .on('mousemove', function(e) {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 10) + 'px';
    }})
    .on('mouseout',  function() {{ tip.style.opacity = '0'; }})
    .on('click',     function(e, d) {{
      if (!e.defaultPrevented) showNodeInfo(d);
    }});
 
  node.attr('transform', function(d) {{
    return 'translate(' + d.x + ',' + d.y + ')';
  }});
 
  svg.call(zoom.transform, d3.zoomIdentity);
}}
 
/* ── Mode B ── */
function renderModeB(nodes, edges) {{
  var W = document.getElementById('canvas').offsetWidth;
  var H = document.getElementById('canvas').offsetHeight;
 
  var idMap = {{}};
  nodes.forEach(function(d) {{ idMap[d.id] = d; }});
 
  var adjEdges = {{}};
  edges.forEach(function(e) {{
    if (!adjEdges[e.source]) adjEdges[e.source] = [];
    if (!adjEdges[e.target]) adjEdges[e.target] = [];
    adjEdges[e.source].push(e);
    adjEdges[e.target].push(e);
  }});
 
  var edgeSel = lg.selectAll('line').data(edges).join('line')
    .attr('class', 'lnk')
    .attr('stroke', '#90a4ae')
    .attr('stroke-width', function(d) {{ return d.width || 1.0; }})
    .attr('x1', function(d) {{ return (idMap[d.source]||{{}}).x||0; }})
    .attr('y1', function(d) {{ return (idMap[d.source]||{{}}).y||0; }})
    .attr('x2', function(d) {{ return (idMap[d.target]||{{}}).x||0; }})
    .attr('y2', function(d) {{ return (idMap[d.target]||{{}}).y||0; }});
 
  var node = ng.selectAll('g').data(nodes, function(d) {{ return d.id; }})
    .join('g').attr('class', 'nd');
 
  node.each(function(d) {{
    var g = d3.select(this);
    var r = d.size || 10;
    if (d.is_mv) {{
      g.append('polygon')
        .attr('points', dpts(r))
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.5);
    }} else {{
      g.append('circle')
        .attr('r', r)
        .attr('fill', d.color).attr('stroke', d.stroke).attr('stroke-width', 1.8);
    }}
  }});
 
  node.append('text').attr('class', 'nlbl')
    .attr('dy', function(d) {{ return -((d.size || 10) + 4); }})
    .attr('font-size', 9)
    .text(function(d) {{
      if (d.is_mv || d.singleton) return '';
      return d.label;
    }});
 
  var dragB = d3.drag()
    .on('start', function(e, d) {{
      svg.on('.zoom', null);
      tip.style.opacity = '0';
      d3.select(this).raise();
    }})
    .on('drag', function(e, d) {{
      var t = d3.zoomTransform(svg.node());
      d.x += e.dx / t.k;
      d.y += e.dy / t.k;
      d3.select(this).attr('transform', 'translate(' + d.x + ',' + d.y + ')');
      var connected = adjEdges[d.id] || [];
      connected.forEach(function(ed) {{
        var s = idMap[ed.source]||{{}}, tg = idMap[ed.target]||{{}};
        d3.select('#edge-' + ed.source + '-' + ed.target)
          .attr('x1', s.x||0).attr('y1', s.y||0)
          .attr('x2', tg.x||0).attr('y2', tg.y||0);
      }});
    }})
    .on('end', function() {{
      if (!lassoOn) svg.call(zoom);
    }});
 
  node.call(dragB);
  edgeSel.attr('id', function(d) {{ return 'edge-' + d.source + '-' + d.target; }});
 
  node.on('mouseover', function(e, d) {{
      tip.style.opacity = '1';
      var members = d.members || [];
      tip.textContent = members.length
        ? members.slice(0,8).join('\\n') + (members.length > 8 ? '\\n...+' + (members.length-8) + ' more' : '')
        : (d.title || d.label || '');
    }})
    .on('mousemove', function(e) {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 10) + 'px';
    }})
    .on('mouseout',  function() {{ tip.style.opacity = '0'; }})
    .on('click',     function(e, d) {{
      if (!e.defaultPrevented) showNodeInfo(d);
    }});
 
  node.attr('transform', function(d) {{
    return 'translate(' + d.x + ',' + d.y + ')';
  }});
 
  var xs = nodes.map(function(d) {{ return d.x || 0; }});
  var ys = nodes.map(function(d) {{ return d.y || 0; }});
  var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
  var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  var sc = Math.min(W / ((x1 - x0) || 1), H / ((y1 - y0) || 1)) * 0.96;
  var tx = W / 2 - cx * sc;
  var ty = H / 2 - cy * sc;
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(sc));
}}
 
/* ── Init ── */
(function() {{
  var nodes = NODES.map(function(d) {{ return Object.assign({{}}, d); }});
  var edges = EDGES.map(function(d) {{ return Object.assign({{}}, d); }});
  if (MODE === 'A') {{ renderModeA(nodes, edges); }}
  else             {{ renderModeB(nodes, edges); }}
}})();
 
/* ── SVG download ── */
function dlSVG() {{
  var original = document.getElementById('net');
  var clone    = original.cloneNode(true);
  clone.querySelectorAll('[stroke-opacity]').forEach(function(el) {{
    el.setAttribute('stroke', el.getAttribute('stroke') || '#000000');
    el.removeAttribute('stroke-opacity');
  }});
  var s = new XMLSerializer();
  var b = new Blob([s.serializeToString(clone)], {{type:'image/svg+xml'}});
  var u = URL.createObjectURL(b);
  var a = document.createElement('a');
  a.href = u; a.download = 'network.svg'; a.click();
  URL.revokeObjectURL(u);
}}
 
window.addEventListener('resize', function() {{
  var nodes = NODES.map(function(d) {{ return Object.assign({{}}, d); }});
  var edges = EDGES.map(function(d) {{ return Object.assign({{}}, d); }});
  lg.selectAll('*').remove();
  eg.selectAll('*').remove();
  ng.selectAll('*').remove();
  if (MODE === 'A') {{ renderModeA(nodes, edges); }}
  else             {{ renderModeB(nodes, edges); }}
}});
</script>
</body>
</html>"""
 
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    mode_str = 'Mode A (Median Joining + MDS)' if mode_a else 'Mode B (Overview + FR layout)'
    logger.info(
        GREEN + BOLD +
        f"[COMPARE] " + NORMAL + f"Network saved ({mode_str}): {output_file}" +
        END_FORMATTING + '\n'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP (only when n_samples <= 100)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_heatmap(
    dist_matrix: np.ndarray,
    sample_names: List[str],
    linkage_matrix: np.ndarray,
    output_file: str,
) -> None:

    """
    Interactive Plotly heatmap of pairwise SNP distances.
    Only generated when n_samples <= 100.
    Color scale adapted to actual data range.
    White background. SVG download via Plotly toolbar.
    """

    n = len(sample_names)

    if n > 100:
        logger.info(
            YELLOW + BOLD + f"[COMPARE] " + NORMAL + f"Heatmap skipped: {n} samples exceeds the 100-sample limit." + END_FORMATTING
        )
        return

    leaf_order     = shc.leaves_list(linkage_matrix)
    ordered_names  = [sample_names[i] for i in leaf_order]
    ordered_matrix = dist_matrix[np.ix_(leaf_order, leaf_order)]

    z_max = float(np.max(ordered_matrix))
    z_min = float(np.min(ordered_matrix[ordered_matrix > 0])) \
            if np.any(ordered_matrix > 0) else 0.0

    fig = go.Figure(data=go.Heatmap(
        z=ordered_matrix,
        x=ordered_names,
        y=ordered_names,
        colorscale='Blues',
        reversescale=True,
        zmin=z_min,
        zmax=z_max,
        hoverongaps=False,
        hovertemplate=(
            '<b>%{y}</b> vs <b>%{x}</b><br>'
            'SNP distance: %{z:.0f}<extra></extra>'
        ),
        colorbar=dict(title='SNP distance'),
    ))

    fig.update_layout(
        title=dict(
            text=f'Pairwise SNP Distance Heatmap ({n} samples)',
            font=dict(size=16, family='Courier New', color='#263238'),
        ),
        paper_bgcolor='#fff',
        plot_bgcolor='#fff',
        font=dict(family='Courier New', size=9, color='#37474f'),
        width=max(700, n * 14),
        height=max(700, n * 14),
        xaxis=dict(tickangle=45),
        modebar_add=['downloadSVG'],
    )

    fig.write_html(output_file, include_plotlyjs='cdn')
    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Heatmap saved: {output_file}" + END_FORMATTING)


# ═══════════════════════════════════════════════════════════════════════════════
# NEWICK OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _write_newick(
    linkage_matrix: np.ndarray,
    sample_names: List[str],
    output_file: str,
) -> None:

    """
    Convert linkage matrix to Newick format using scipy's to_tree.
    Uses UPGMA (average linkage) — standard for epidemiological SNP trees.
    """

    tree = shc.to_tree(linkage_matrix, False)

    def build_newick(node, newick: str, parent_dist: float, leaf_names: List[str]) -> str:
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parent_dist - node.dist:.6f}{newick}"
        if len(newick) > 0:
            newick = f"):{(parent_dist - node.dist) / 2:.6f}{newick}"
        else:
            newick = ");"
        newick = build_newick(node.get_left(),  newick, node.dist, leaf_names)
        newick = build_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
        return f"({newick}"

    nwk = build_newick(tree, "", tree.dist, sample_names)

    with open(output_file, 'w') as f:
        f.write(nwk)

    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Newick file saved: {output_file}" + END_FORMATTING + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# RDF FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def matrix_to_rdf(snp_df: pl.DataFrame, output_file: str) -> None:
    """
    Write the SNP matrix in RDF format for downstream tools.
    """
    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in snp_df.columns if c not in index_cols]

    snp_number = snp_df.height
    snp_list   = snp_df['Position'].to_list()
    pos_list   = [p.split('|')[2] for p in snp_list]

    with open(output_file, 'w') as f:
        f.write("  ;1.0\n")
        f.write(" ;".join(pos_list) + " ;\n")
        f.write(("10;" * snp_number) + "\n")

        # Sample rows
        values_matrix = snp_df.select(sample_cols).to_numpy().astype(float)
        for i, sample in enumerate(sample_cols):
            f.write(f">{sample};1;;;;;;;\n")
            row_str = "".join(
                '1' if v >= 0.5 else '0'
                for v in values_matrix[:, i]
            )
            f.write(row_str + "\n")

        # Reference row (all zeros)
        f.write(">REF;1;;;;;;;\n")
        f.write("0" * snp_number + "\n")

    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"RDF file saved: {output_file}" + END_FORMATTING + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def ddtb_compare(
    snp_matrix: pl.DataFrame,
    output_path: str,
    distance: int           = 15,
    genomic_distance: int   = 50,
    max_workers: int        = 1,
) -> None:

    """
    Main comparison function. Given a binarised SNP matrix, computes
    pairwise distances and generates all output files and visualisations.

    Outputs
    -------
    <output_path>.snp.tsv                : pairwise SNP distance matrix
    <output_path>.snp.pairwise.tsv       : pairwise distances (long format)
    <output_path>.hamming.tsv            : normalised Hamming distance matrix
    <output_path>.nwk                    : Newick tree (UPGMA, avg linkage)
    <output_path>.rdf                    : RDF binary matrix
    <output_path>.common.tsv            : common SNPs per cluster
    <output_path>.clusters_<d>.tsv      : cluster assignments at distance d
    <output_path>.dendrogram.html        : interactive Plotly dendrogram
    <output_path>.heatmap.html           : interactive Plotly heatmap
    <output_path>.network.html           : interactive Pyvis genomic network

    Parameters
    ----------
    snp_matrix       : pl.DataFrame from revised_df (binarised, '0'/'0.5'/'1')
    output_path      : base path for all output files (no extension)
    distance         : SNP threshold for cluster assignment and collapse
    genomic_distance : SNP threshold for network edges (default 100)
    max_workers      : threads for distance computation
    """

    logger.info(BOLD + CYAN + "[COMPARE] " + NORMAL + f"Starting ddtb_compare" + END_FORMATTING)

    # ── Extract and convert matrix ────────────────────────────────────────────
    # Convert string values to float: '0' → 0.0, '0.5' → 0.5, '1' → 1.0
    # Any non-numeric value defaults to 0.0 (safe fallback)
    index_cols  = ['Position', 'N', 'Samples']
    sample_cols = [c for c in snp_matrix.columns if c not in index_cols]
    positions   = snp_matrix['Position'].to_list()

    logger.info(
        GREEN + BOLD + f"[COMPARE] " + NORMAL + f"{snp_matrix.height} positions × {len(sample_cols)} samples" + END_FORMATTING + '\n'
    )

    # Build float matrix (n_positions × n_samples)
    raw = snp_matrix.select(sample_cols).to_numpy()
    float_matrix = np.zeros(raw.shape, dtype=float)
    for j in range(raw.shape[1]):
        for i in range(raw.shape[0]):
            try:
                float_matrix[i, j] = float(raw[i, j])
            except (ValueError, TypeError):
                float_matrix[i, j] = 0.0

    matrix_T = float_matrix.T  # (n_samples × n_positions) for row-wise ops

    # ── Pairwise distance matrix ──────────────────────────────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL + "Computing pairwise distances..." + END_FORMATTING)
    dist_matrix, pairwise_df = compute_distance_matrix(
        float_matrix, sample_cols, max_workers=max_workers
    )

    # SNP distance matrix (integer distances)
    snp_dist_int = dist_matrix.astype(int)
    snp_dist_df = pl.DataFrame(
        {sample_cols[i]: snp_dist_int[:, i].tolist() for i in range(len(sample_cols))}
    ).insert_column(0, pl.Series('sample', sample_cols))

    snp_dist_df.write_csv(output_path + ".snp.tsv", separator='\t')
    pairwise_df.write_csv(output_path + ".snp.pairwise.tsv", separator='\t')
    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"SNP distance matrix: {output_path}.snp.tsv" + END_FORMATTING)

    # Normalised Hamming distance matrix
    n_pos = float_matrix.shape[0]
    hamming_norm = dist_matrix / n_pos if n_pos > 0 else dist_matrix
    hamming_df = pl.DataFrame(
        {sample_cols[i]: hamming_norm[:, i].tolist() for i in range(len(sample_cols))}
    ).insert_column(0, pl.Series('sample', sample_cols))
    hamming_df.write_csv(output_path + ".hamming.tsv", separator='\t')
    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Hamming distance matrix: {output_path}.hamming.tsv" + END_FORMATTING + '\n')

    # ── Hierarchical clustering (UPGMA / average linkage) ────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL +  f"Building hierarchical clustering..." + END_FORMATTING)
    condensed = ssd.squareform(dist_matrix)
    linkage_matrix = shc.linkage(condensed, method='average')

    # ── Cluster assignment ────────────────────────────────────────────────────
    clusters = _assign_clusters(dist_matrix, sample_cols, distance)
    clusters.write_csv(
        output_path + f".clusters_{distance}.tsv", separator='\t'
    )
    logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Cluster assignments: {output_path}.clusters_{distance}.tsv" + END_FORMATTING + '\n')

    # ── Newick ────────────────────────────────────────────────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL + "Writing Newick file..." + END_FORMATTING)
    _write_newick(linkage_matrix, sample_cols, output_path + ".nwk")

    # ── RDF ───────────────────────────────────────────────────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL + "Writing RDF file..." + END_FORMATTING)
    matrix_to_rdf(snp_matrix, output_path + ".rdf")

    # ── Common SNPs per cluster ───────────────────────────────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL + "Computing common SNPs per cluster..." + END_FORMATTING)
    cluster_df, node_snps = compute_cluster_snps(snp_matrix, dist_matrix, sample_cols, linkage_matrix, distance, output_path + ".common.tsv")

    # ── Visualisations ────────────────────────────────────────────────────────
    logger.info(CYAN + BOLD + "[COMPARE] " + NORMAL + "Building visualisations..." + END_FORMATTING)

    _build_dendrogram(
        linkage_matrix=linkage_matrix,
        sample_names=sample_cols,
        node_snps=node_snps,
        dist_matrix=dist_matrix,
        distance_threshold=distance,
        output_file=output_path + ".dendrogram.html",
    )

    _build_heatmap(
        dist_matrix=dist_matrix,
        sample_names=sample_cols,
        linkage_matrix=linkage_matrix,
        output_file=output_path + ".heatmap.html",
    )

    _build_network(
    dist_matrix=dist_matrix,
    sample_names=sample_cols,
    genomic_threshold=genomic_distance,
    collapse_threshold=distance,
    output_file=output_path + ".network.html",
    snp_matrix_T=matrix_T,
    )

    logger.info(BOLD + GREEN + "[COMPARE] All outputs complete." + END_FORMATTING + '\n')


if __name__ == '__main__':

    args = get_arguments()

    output_dir = os.path.abspath(args.output)
    group_name = output_dir.split('/')[-1]
    check_create_dir(output_dir)
    # LOGGING
    # Create log file with date and time
    right_now = str(datetime.datetime.now())
    right_now_full = "_".join(right_now.split(" "))
    log_filename = group_name + "_" + right_now_full + ".log"
    log_folder = os.path.join(output_dir, 'Logs')
    check_create_dir(log_folder)
    log_full_path = os.path.join(log_folder, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(message)s')

    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("#################### COMPARE SNPS #########################")
    logger.info(args)

    group_compare = os.path.join(output_dir, group_name)
    compare_snp_matrix = group_compare + ".tsv"
    input_dir = os.path.abspath(args.input_dir)

    out_variant_dir = os.path.join(input_dir, "Variants")
    out_stats_dir = os.path.join(input_dir, "Stats")
    out_stats_coverage_dir = os.path.join(out_stats_dir, "Coverage")  # subfolder


    logger.info("\n" + BLUE + BOLD + "STARTING COMPARISON IN GROUP: " + NORMAL + group_name + END_FORMATTING + "\n")


    # ── Sample list ──────────────────────────────────────────────────────────

    if args.sample_list:
        sample_file = os.path.abspath(args.sample_list)
        if not os.path.exists(sample_file):
            logger.error(
                '\n' + RED + BOLD + f"[SAMPLES] File not found: {sample_file}" + END_FORMATTING + '\n'
            )
            sys.exit(1)
        else:
            with open(sample_file) as f:
                sample_list = {x.strip() for x in f if x.strip()}
            logger.info(YELLOW + BOLD + f"[MODE] " + NORMAL + f"Targeted ({len(sample_list)} samples)" + END_FORMATTING + '\n')
    else:
        logger.info(YELLOW + BOLD + "[MODE] " + NORMAL + f"Full dataset" + END_FORMATTING + '\n')
        sample_list = False

    # ── Threads ──────────────────────────────────────────────────────────────
    max_workers = args.threads if args.threads else min(32, os.cpu_count() or 1)

    # ── Complex/hotspot filtering (active by default; --complex disables it) ──
    apply_complex = not args.complex
    remove_faulty = not args.no_remove_faulty


    if args.only_compare == False:

        today = str(datetime.date.today())
        check_create_dir(output_dir)
        folder_compare = today + "_" + group_name
        path_compare = os.path.join(output_dir, folder_compare)
        check_create_dir(path_compare)
        full_path_compare = os.path.join(path_compare, group_name)

        compare_snp_matrix_recal = full_path_compare + ".revised.FULL.tsv"
        compare_snp_matrix_recal_intermediate = full_path_compare + ".revised_intermediate.tsv"
        compare_snp_matrix_recal_mpileup = full_path_compare + ".revised_intermediate_vcf.tsv"
        compare_snp_matrix_INDEL_intermediate = full_path_compare + ".revised_INDEL_intermediate.tsv"
        compare_only_snps = full_path_compare + ".revised.SNPs.tsv"
        complex_file = full_path_compare + ".complex_hotspot.tsv"
        symbol_file = full_path_compare + '_symbol_lowcov.tsv'
        compare_final_snps = full_path_compare + ".revised.FINAL.tsv"

        # ── Create intermediate matrix ───────────────────────────────────────────

        prior = datetime.datetime.now()

        recalibrated_snp_matrix_intermediate, complex_pos_dict = ddbb_create_intermediate(
            variant_dir=out_variant_dir,
            coverage_dir=out_stats_coverage_dir,
            min_total_depth=args.min_total_depth,
            min_cov=args.min_cov,
            min_freq_discard=args.min_freq_discard,
            min_freq_include=args.min_freq_include,
            apply_complex = apply_complex,
            complex_output_tsv=complex_file if apply_complex else None,
            window_size = args.window,
            max_variants_window = args.variant_window,
            samples=sample_list,
            max_workers=max_workers,  # None = min(32, cpu_count); adjust according to machine
        )

        if recalibrated_snp_matrix_intermediate.is_empty():
            logger.error(RED + BOLD + "[FATAL] Intermediate matrix is empty. Aborting." + END_FORMATTING)
            sys.exit(1)


        # ── Remove SNPs from BED file ────────────────────────────────────────────

        if args.remove_bed:
            recalibrated_snp_matrix_intermediate = remove_bed_positions(
                recalibrated_snp_matrix_intermediate, args.remove_bed, full_path_compare)

        recalibrated_snp_matrix_intermediate.write_csv(
            compare_snp_matrix_recal_intermediate, separator="\t"
        )

        logger.info(
            GREEN + BOLD + "[COMPARE] " + NORMAL + f"Intermediate matrix saved to: {compare_snp_matrix_recal_intermediate}" + END_FORMATTING + '\n'
        )

        after = datetime.datetime.now()
        logger.info("Done with function ddbb_create_intermediate: %s" %
                     (after - prior) + "\n")


        # ── Recalibrate intermediate with VCF ────────────────────────────────────

        prior = datetime.datetime.now()

        recalibrated_snp_matrix_mpileup = recalibrate_ddbb_vcf_intermediate(
            snp_matrix=recalibrated_snp_matrix_intermediate,
            variant_folder=out_variant_dir,
            min_total_depth=args.min_total_depth,
            min_cov=args.min_cov,
            min_freq_include=args.min_freq_include,
            min_freq_discard=args.min_freq_discard,
            max_workers=max_workers,
            # complex_positions_per_sample=complex_pos_dict if apply_complex else None,
            )

        recalibrated_snp_matrix_mpileup.write_csv(compare_snp_matrix_recal_mpileup, separator="\t")

        after = datetime.datetime.now()
        logger.info("Done with recalibration vcf: %s" %
                     (after - prior)  + "\n")


        # ── Remove SNPs located within INDELs ────────────────────────────────────

        prior = datetime.datetime.now()

        compare_snp_matrix_INDEL_intermediate_df = remove_position_range(
            recalibrated_snp_matrix_mpileup)
        compare_snp_matrix_INDEL_intermediate_df.write_csv(compare_snp_matrix_INDEL_intermediate, separator="\t")

        after = datetime.datetime.now()
        logger.info(("Done with function remove_position_range in: %s" %
               (after - prior) + "\n"))


        # ── Extract all low coverage o not covered positions ─────────────────────

        prior = datetime.datetime.now()

        symbol_lowcov = extract_lowcov(compare_snp_matrix_INDEL_intermediate_df, min_freq_include=args.min_freq_include) # It is made by the INDEL_intermediate.tsv, taking 0 and 1 into account, can also be made with intermediate.highfreq.tsv
        symbol_lowcov.write_csv(symbol_file, separator='\t')

        after = datetime.datetime.now()
        print(("Done with function extract_lowcov in: %s" %
               (after - prior) + "\n"))


        # ── Clean all faulty positions and samples => Revised table ──────────────

        prior = datetime.datetime.now()

        full_df, snps_df = revised_df(compare_snp_matrix_INDEL_intermediate_df, 
                                      path_compare, 
                                      symbol_lowcov=symbol_lowcov,
                                      min_freq_include=args.min_freq_include, 
                                      min_freq_discard=args.min_freq_discard, 
                                      min_threshold_discard_uncov_pos=args.min_threshold_discard_uncov_pos, 
                                      min_threshold_discard_htz_pos=args.min_threshold_discard_htz_pos, 
                                      min_threshold_discard_all_pos=args.min_threshold_discard_all_pos, 
                                      min_threshold_discard_uncov_sample=args.min_threshold_discard_uncov_sample, 
                                      min_threshold_discard_htz_sample=args.min_threshold_discard_htz_sample, 
                                      min_threshold_discard_all_sample=args.min_threshold_discard_all_sample, 
                                      remove_faulty=remove_faulty,
                                      )

        full_df.write_csv(compare_snp_matrix_recal, separator='\t')
        snps_df.write_csv(compare_only_snps, separator='\t')

        final_snps_df = pl.concat([snps_df.filter(pl.col('N') != (len(snps_df.columns) - 3)), snps_df.filter(pl.col('N') == (len(snps_df.columns) - 3)).head(1)]).sort('N', descending=False)
        final_snps_df.write_csv(compare_final_snps, separator='\t')

        after = datetime.datetime.now()
        print(("Done with function revised_df in: %s" %
               (after - prior) + "\n"))


        # ── Annotated SNPs from BED file (genes or positions of interest) ────────

        if args.extract_bed:
            prior = datetime.datetime.now()

            annotated_snps_final = extract_bed_positions(
                full_df, args.extract_bed, full_path_compare
                )

            after = datetime.datetime.now()
            print(("Done with function extract_bed_positions in: %s" %
                (after - prior) + "\n"))


        # ── Matrix to pairwise and mwk ───────────────────────────────────────────

        prior = datetime.datetime.now()

        ddtb_compare(
            snp_matrix=final_snps_df,
            output_path=full_path_compare,
            distance=args.distance,
            genomic_distance=args.genomic,
            max_workers=max_workers,
            )

        after = datetime.datetime.now()
        print(("Done with function ddtb_compare in: %s" %
               (after - prior) + "\n"))


    else:
        prior = datetime.datetime.now()

        path_input = Path(args.input_dir)

        if path_input.is_file():
            compare_final_snps = path_input
        else:
            compare_final_snps = list(path_input.glob('*.revised.SNPs.tsv'))[0]

        # Load existing SNP matrix from file
        if not os.path.exists(compare_final_snps):
            logger.error(RED + BOLD + f"[FATAL] " + NORMAL + f"SNP matrix not found: {compare_final_snps}" + END_FORMATTING + '\n')
            sys.exit(1)

        snps_df_compare = pl.read_csv(compare_final_snps, separator='\t', infer_schema_length=0)
        logger.info(GREEN + BOLD + f"[COMPARE] " + NORMAL + f"Loaded existing matrix: {compare_final_snps}" + END_FORMATTING + '\n')

        ddtb_compare(
            snp_matrix=snps_df_compare,
            output_path=args.output,
            distance=args.distance,
            genomic_distance=args.genomic,
            max_workers=max_workers,
        )

        after = datetime.datetime.now()
        print(("Done with function ddtb_compare in: %s" %
               (after - prior) + "\n"))


    logger.info("\n" + MAGENTA + BOLD + "COMPARING FINISHED IN GROUP: " + NORMAL +
                    group_name + END_FORMATTING + "\n")

    logger.info("\n" + MAGENTA + BOLD +
                    "#####END OF PIPELINE AUTOSNIPPY ANALYSIS#####" + END_FORMATTING)