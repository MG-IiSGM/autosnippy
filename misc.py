
import os
import sys
import re
import subprocess
import shutil
import logging
import datetime
import pandas as pd
import numpy as np
from statistics import mean
from pandarallel import pandarallel
import concurrent.futures


logger = logging.getLogger()

# COLORS AND AND FORMATTING
"""
http://ozzmaker.com/add-colour-to-text-in-python/
The above ANSI escape code will set the text colour to bright green. The format is;
\033[  Escape code, this is always the same
1 = Style, 1 for normal.
32 = Text colour, 32 for bright green.
40m = Background colour, 40 is for black.
"""
END_FORMATTING = '\033[0m'
WHITE_BG = '\033[0;30;47m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
BLUE = '\033[34m'
CYAN = '\033[36m'
YELLOW = '\033[93m'
DIM = '\033[2m'


def check_file_exists(file_name):
    """
        Check file exist and is not 0 Kb, if not program exit.
    """
    file_info = os.stat(
        file_name)  # Retrieve the file info to check if has size > 0

    if not os.path.isfile(file_name) or file_info.st_size == 0:
        logger.info(RED + BOLD + "File: %s not found or empty\n" %
                    file_name + END_FORMATTING)
        sys.exit(1)
    return os.path.isfile(file_name)


def check_remove_file(file_name):
    """
    Check file exist and remove it.
    """
    if os.path.exists(file_name):
        os.remove(file_name)


def import_to_pandas(file_table, header=False, sep='\t'):
    if header == False:
        # exclude first line, exclusive for vcf outputted by PipelineTB
        dataframe = pd.read_csv(file_table, sep=sep, skiprows=[0], header=None)
    else:
        # Use first line as header
        dataframe = pd.read_csv(file_table, sep=sep, header=0)

    return dataframe


def extract_sample(R1_file, R2_file):
    """
    Extract sample from R1, R2 files.
    """
    basename_R1 = os.path.basename(R1_file)
    basename_R2 = os.path.basename(R2_file)

    sample_name_R = os.path.commonprefix([basename_R1, basename_R2])

    long_suffix = re.search('_S.*', sample_name_R)
    short_suffix = re.search('_R.*', sample_name_R)
    bar_suffix = re.search('_$', sample_name_R)
    dot_suffix = re.search('.R$', sample_name_R)

    if long_suffix:
        match = long_suffix.group()
        sample_name = sample_name_R.split(match)[0]
    elif short_suffix:
        match = short_suffix.group()
        sample_name = sample_name_R.split(match)[0]
    elif bar_suffix:
        match = bar_suffix.group()
        sample_name = sample_name_R.rstrip("_")
    elif dot_suffix:
        match = dot_suffix.group()
        sample_name = sample_name_R.rstrip(".R")
    else:
        sample_name = sample_name_R

    return sample_name


def check_create_dir(path):
    #exists = os.path.isfile(path)
    #exists = os.path.isdir(path)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def execute_subprocess(cmd, isShell=False):
    """
    https://crashcourse.housegordon.org/python-subprocess.html
    https://docs.python.org/3/library/subprocess.html 
    Execute and handle errors with subprocess, outputting stderr instead of the subprocess CalledProcessError
    """

    logger.debug("")
    logger.debug(cmd)

    if cmd[0] == "java":
        prog = cmd[2].split("/")[-1] + " " + cmd[3]
        param = cmd[4:]
    elif cmd[0] == "samtools" or cmd[0] == "bwa" or cmd[0] == "gatk":
        prog = " ".join(cmd[0:2])
        param = cmd[3:]
    else:
        prog = cmd[0]
        param = cmd[1:]

    try:
        command = subprocess.run(
            cmd, shell=isShell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if command.returncode == 0:
            logger.debug(
                GREEN + DIM + "Program %s successfully executed" % prog + END_FORMATTING)
        else:
            logger.info(RED + BOLD + "Command %s FAILED\n" % prog + END_FORMATTING
                        + BOLD + "WITH PARAMETERS: " +
                        END_FORMATTING + " ".join(param) + "\n"
                        + BOLD + "EXIT-CODE: %d\n" % command.returncode +
                        "ERROR:\n" + END_FORMATTING + command.stderr.decode().strip())
        logger.debug(command.stdout)
        logger.debug(command.stderr.decode().strip())
    except OSError as e:
        sys.exit(RED + BOLD + "failed to execute program '%s': %s" % (prog,
                                                                      str(e)) + END_FORMATTING)


def extract_read_list_legacy(input_dir):
    """
    Search files in a directory sort by name and extract comon name of R1 and R2
    with extract_sample() function
    190615 - Limit only parent folder, not subdirectories
    """
    input_dir = os.path.abspath(input_dir)
    r1_list = []
    r2_list = []
    for root, _, files in os.walk(input_dir):
        if root == input_dir:  # This only apply to parent folder, not subdirectories
            for name in files:
                filename = os.path.join(root, name)
                is_fasta = re.match(r'.*\.f(ast)*[aq](\.gz)*', name)
                r1 = re.match(
                    r'.*(_R1_|_1|_1_|_R1).*\.f(ast)*[aq](\.gz)*$', name)
                r2 = re.match(
                    r'.*(_R2_|_2|_2_|_R2).*\.f(ast)*[aq](\.gz)*$', name)
                if is_fasta:
                    if r1:
                        r1_list.append(filename)
                    elif r2:
                        r2_list.append(filename)
                    else:
                        logger.info(
                            RED + "ERROR, file is not R1 nor R2" + END_FORMATTING)
                        sys.exit(1)
    r1_list = sorted(r1_list)
    r2_list = sorted(r2_list)
    return r1_list, r2_list


def extract_read_list(input_dir):
    """
    Search files in a directory sort by name and extract comon name of R1 and R2
    with extract_sample() function
    190615 - Limit only parent folder, not subdirectories
    """
    input_dir = os.path.abspath(input_dir)
    all_fasta = []
    r1_list = []
    r2_list = []
    for root, _, files in os.walk(input_dir):
        if root == input_dir:  # This only apply to parent folder, not subdirectories
            for name in files:
                filename = os.path.join(root, name)
                is_fasta = re.match(r'.*\.f(ast)*[aq](\.gz)*', filename)
                if is_fasta:
                    all_fasta.append(filename)
    all_fasta = sorted(all_fasta)
    samples = [os.path.basename(fasta).split('_')[0] for fasta in all_fasta]
    remove_sample = [fasta for fasta in samples if samples.count(fasta) == 1]   # Remove unpaired samples
    for fasta in all_fasta:
        for pattern in remove_sample:
            if re.search(pattern, fasta, flags=0): all_fasta.remove(fasta)
    for index, fasta_file in enumerate(all_fasta):
        if index % 2 == 0:
            r1_list.append(fasta_file)
        elif index % 1 == 0:
            r2_list.append(fasta_file)

    r1_list = sorted(r1_list)
    r2_list = sorted(r2_list)
    return r1_list, r2_list, remove_sample


def return_codon_position(number):
    position = number % 3
    if position == 0:
        position = 3
    logger.info("number= %s, pos= %s" % (number, position))


def file_to_list(file_name):
    list_F = []
    file_name_abs = os.path.abspath(file_name)
    with open(file_name_abs, "r") as f:
        for line in f:
            list_F.append(line.strip())
    return list_F


def calculate_cov_stats(file_cov):
    sample = file_cov.split("/")[-1].split(".")[0]
    df = pd.read_csv(file_cov, sep="\t", names=["#CHROM", "POS", "COV"])
    unmapped_pos = len(df.POS[df.COV == 0].tolist())
    pos_0_10 = len(df.POS[(df.COV > 0) & (df.COV <= 10)].tolist())
    pos_10_20 = len(df.POS[(df.COV > 10) & (df.COV <= 20)].tolist())
    pos_high20 = len(df.POS[(df.COV > 20)].tolist())
    pos_high50 = len(df.POS[(df.COV > 50)].tolist())
    pos_high100 = len(df.POS[(df.COV >= 100)].tolist())
    pos_high500 = len(df.POS[(df.COV >= 500)].tolist())
    pos_high1000 = len(df.POS[(df.COV >= 1000)].tolist())
    total_pos = df.shape[0]
    unmapped_prop = "%.2f" % ((unmapped_pos/total_pos)*100)
    prop_0_10 = "%.2f" % ((pos_0_10/total_pos)*100)
    prop_10_20 = "%.2f" % ((pos_10_20/total_pos)*100)
    prop_high20 = "%.2f" % ((pos_high20/total_pos)*100)
    prop_high50 = "%.2f" % ((pos_high50/total_pos)*100)
    prop_high100 = "%.2f" % ((pos_high100/total_pos)*100)
    prop_high500 = "%.2f" % ((pos_high500/total_pos)*100)
    prop_high1000 = "%.2f" % ((pos_high1000/total_pos)*100)

    mean_cov = "%.2f" % (df.COV.mean())

    return sample, mean_cov, unmapped_prop, prop_0_10, prop_10_20, prop_high20, prop_high50, prop_high100, prop_high500, prop_high1000


def obtain_group_cov_stats(directory, group_name, removed_samples):
    directory_path = os.path.abspath(directory)
    samples_to_skip = removed_samples
    previous_stat = False

    output_group_name = group_name + ".coverage.summary.tab"
    output_file = os.path.join(directory_path, output_group_name)

    if os.path.exists(output_file):
        previous_stat = True
        df_stat = pd.read_csv(output_file, sep="\t")
        samples_to_skip = df_stat["#SAMPLE"].tolist()
        logger.debug("Skipped samples for coverage calculation:" +
                     (",").join(samples_to_skip))

    columns = ["#SAMPLE", "MEAN_COV", "UNMAPPED_PROP", "COV1-10X",
               "COV10-20X", "COV>20X", "COV>50X", "COV>100X", "COV>500X", "COV>1000X"]

    files_list = []

    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.cov'):
                filename = os.path.join(root, name)
                sample = name.split(".")[0]
                #df[columns] = df.apply(calculate_cov_stats(filename), axis=1, result_type="expand")
                if not sample in samples_to_skip:
                    files_list.append(filename)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        dfs = executor.map(calculate_cov_stats, files_list)
    df = pd.DataFrame(dfs, columns=columns)

    if previous_stat:
        df = pd.concat([df_stat, df], ignore_index=True, sort=True)
        df.to_csv(output_file, sep="\t", index=False)
    else:
        df.to_csv(output_file, sep="\t", index=False)


def obtain_group_cov_stats_L(directory, group_name):
    directory_path = os.path.abspath(directory)

    output_group_name = group_name + ".coverage.summary.tab"
    output_file = os.path.join(directory_path, output_group_name)

    with open(output_file, "w+") as outfile:
        outfile.write("#SAMPLE" + "\t" + "MEAN_COV" + "\t" + "UNMAPPED_PROP" + "\t" + "COV1-10X" + "\t" + "COV10-20X" +
                      "\t" + "COV>20X" + "\t" + "COV>50X" + "\t" + "COV>100X" + "\t" + "COV>500X" + "\t" + "COV>1000X" + "\n")
        for root, _, files in os.walk(directory_path):
            for name in files:
                filename = os.path.join(root, name)
                file_name_cov = os.path.basename(filename)
                sample = file_name_cov.split(".")[0]
                if filename.endswith(".cov") and (os.path.getsize(filename) > 0):
                    coverage_stats = calculate_cov_stats(filename)
                    outfile.write(sample + "\t" +
                                  ("\t").join(coverage_stats) + "\n")


def extract_snp_count(output_dir, sample):
    sample = str(sample)
    if '.' in sample:
        sample = sample.split('.')[0]
    variants_folder = os.path.join(output_dir, 'Variants')
    raw_var_folder = os.path.join(variants_folder, sample)
    filename = os.path.join(raw_var_folder, "snps.all.ivar.tsv")

    if os.path.exists(filename):
        df = pd.read_csv(filename, sep="\t")
        df = df.drop_duplicates(subset=['POS', 'REF', 'ALT'], keep="first")
        high_quality_snps = df["POS"][(df.ALT_DP >= 20) &
                                      (df.ALT_FREQ >= 0.8) &
                                      (df.TYPE == 'snp')].tolist()
        htz_snps = df["POS"][(df.ALT_DP >= 20) &
                             (df.ALT_FREQ < 0.8) &
                             (df.ALT_FREQ >= 0.2) &
                             (df.TYPE == 'snp')].tolist()
        indels = df["POS"][(df.ALT_DP >= 20) &
                           (df.ALT_FREQ >= 0.8) &
                           ((df.TYPE == 'ins') | (df.TYPE == 'del'))].tolist()
        return (len(high_quality_snps), len(htz_snps), len(indels))
    else:
        logger.debug("FILE " + filename + " NOT FOUND")
        return None


def extract_mapped_reads(output_dir, sample):
    sample = str(sample)
    if '.' in sample:
        sample = sample.split('.')[0]
    stats_folder = os.path.join(output_dir, 'Stats')
    bamstats_folder = os.path.join(stats_folder, 'Bamstats')
    filename = os.path.join(bamstats_folder, sample + ".bamstats")

    if os.path.exists(filename):
        reads_mapped = 0
        mappep_percentage = 0
        properly_paired = 0
        paired_percentage = 0
        with open(filename, 'r') as f:
            logger.debug('File bamstat: {}'.format(filename))
            for line in f:
                if 'mapped' in line and '%' in line:
                    reads_mapped = line.split(" ")[0]
                    mappep_percentage = line.split("(")[-1].split("%")[0]
                elif 'properly paired' in line:
                    properly_paired = line.split(" ")[0]
                    paired_percentage = line.split("(")[-1].split("%")[0]

        # logger.debug((",").join(
        #     [reads_mapped, mappep_percentage, properly_paired, paired_percentage]))

        # logger.debug(len([x for x in [reads_mapped, mappep_percentage,
        #                               properly_paired, paired_percentage] if x != 0]))

        if len([x for x in [reads_mapped, mappep_percentage, properly_paired, paired_percentage] if x != 0]):
            return int(reads_mapped), float(mappep_percentage), int(properly_paired), float(paired_percentage)
        else:
            return 0, 0, 0, 0
    else:
        logger.info("FILE " + filename + " NOT FOUND")
        return None


def extract_n_consensus(output_dir, sample):
    sample = str(sample)
    if '.' in sample:
        sample = sample.split('.')[0]
    consensus_folder = os.path.join(output_dir, 'Variants')
    consensus_folder_sample = os.path.join(consensus_folder, sample)
    filename = os.path.join(consensus_folder_sample, "snps.consensus.fa")

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            content_list = content.split('\n')
            #sample_fq = content_list[0].strip(">")
            # In case fasta is in several lines(not by default)
            sequence = ("").join(content_list[1:]).strip()
            all_N = re.findall(r'N+', sequence)
            if all_N:
                leading_N = re.findall(r'^N+', sequence)
                tailing_N = re.findall(r'N+$', sequence)
                length_N = [len(x) for x in all_N]
                individual_N = [x for x in length_N if x == 1]
                mean_length_N = mean(length_N)
                sum_length_N = sum(length_N)
                total_perc_N = sum_length_N / len(sequence) * 100
                return(len(all_N), len(individual_N), len(leading_N), len(tailing_N), sum_length_N, total_perc_N, mean_length_N)
            else:
                return(0, 0, 0, 0, 0, 0, 0)

    else:
        logger.info("FILE " + filename + " NOT FOUND")
        return None


def obtain_overal_stats(output_dir, group):
    pandarallel.initialize()
    samples_to_skip = []
    previous_stat = False

    stat_folder = os.path.join(output_dir, 'Stats')
    overal_stat_file = os.path.join(stat_folder, group + ".overal.stats.tab")

    columns = ['#SAMPLE', 'MEAN_COV', 'UNMAPPED_PROP', 'COV1-10X', 'COV10-20X', 'COV>20X', 'COV>50X', 'COV>100X',
               'COV>500X', 'COV>1000X']

    if os.path.exists(overal_stat_file):
        previous_stat = True
        df_stat = pd.read_csv(overal_stat_file, sep="\t")
        samples_to_skip = df_stat["#SAMPLE"].tolist()
        logger.debug("Skipped samples for coverage calculation:" +
                     (",").join(samples_to_skip))

    for root, _, files in os.walk(stat_folder):
        for name in files:
            if name.endswith('coverage.summary.tab'):
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep="\t")
                df = df[~df["#SAMPLE"].isin(samples_to_skip)]
                if df.shape[0] > 0:
                    df[['HQ_SNP', 'HTZ_SNP', 'INDELS']] = df.parallel_apply(lambda x: extract_snp_count(
                        output_dir, x['#SAMPLE']), axis=1, result_type="expand")
                    df[['mapped_reads', 'perc_mapped', 'paired_mapped', 'perc_paired']] = df.parallel_apply(
                        lambda x: extract_mapped_reads(output_dir, x['#SAMPLE']), axis=1, result_type="expand")
                    df[['N_groups', 'N_individual', 'N_leading', 'N_tailing', 'N_sum_len', 'N_total_perc', 'N_mean_len']] = df.parallel_apply(
                        lambda x: extract_n_consensus(output_dir, x['#SAMPLE']), axis=1, result_type="expand")

    if previous_stat:
        df = pd.concat([df_stat, df], ignore_index=True, sort=True)
        df = df[columns + [col for col in df.columns if col != "#SAMPLE" and col != "MEAN_COV" and col != "UNMAPPED_PROP" and col != "COV1-10X" and col != "COV10-20X" and col != "COV>20X" and col != "COV>50X" and col !=
                           "COV>100X" and col != "COV>500X" and col != "COV>1000X"]]
        df.to_csv(overal_stat_file, sep="\t", index=False)
    else:
        df = df[columns + [col for col in df.columns if col != "#SAMPLE" and col != "MEAN_COV" and col != "UNMAPPED_PROP" and col != "COV1-10X" and col != "COV10-20X" and col != "COV>20X" and col != "COV>50X" and col !=
                           "COV>100X" and col != "COV>500X" and col != "COV>1000X"]]
        df.to_csv(overal_stat_file, sep="\t", index=False)


def edit_sample_list(file_list, sample_list):
    with open(file_list, 'r') as f:
        content = f.read()
        content_list = content.split('\n')
        while '' in content_list:
            content_list.remove('')

    with open(file_list, 'w+') as fout:
        for line in content_list:
            if line not in sample_list:
                fout.write(line + "\n")


def remove_low_quality(output_dir, mean_cov=20, min_coverage=30, min_hq_snp=8, type_remove='Uncovered'):
    right_now = str(datetime.datetime.now())
    right_now_full = "_".join(right_now.split(" "))
    output_dir = os.path.abspath(output_dir)
    uncovered_dir = os.path.join(output_dir, type_remove)  # Uncovered or Mixed
    uncovered_dir_variants = os.path.join(uncovered_dir, "Variants")

    check_create_dir(uncovered_dir)

    uncovered_samples = []

    for root, _, files in os.walk(output_dir):
        # Any previous file created except for Table for mixed samples
        # and Species for both uncovered and mixed
        if root.endswith('Stats'):
            for name in files:
                filename = os.path.join(root, name)
                if name.endswith('overal.stats.tab'):
                    coverage_stat_file = filename
                    stats_df = pd.read_csv(coverage_stat_file, sep="\t")
                    stats_df = stats_df.fillna(0)

                    # Handle repeated names
                    stats_df['HQ_SNP'] = stats_df['HQ_SNP'].astype(str)
                    def f(x): return x if x.replace(
                        '.', '', 1).isdigit() else max(x.strip("()").split(","))
                    stats_df['HQ_SNP'] = stats_df.apply(
                        lambda x: f(x.HQ_SNP), axis=1)

                    stats_df['HQ_SNP'] = stats_df['HQ_SNP'].astype(float)
                    uncovered_samples = stats_df["#SAMPLE"][(stats_df['MEAN_COV'] <= mean_cov) | (stats_df["UNMAPPED_PROP"] >= min_coverage) | (
                        stats_df["HQ_SNP"] < min_hq_snp)].tolist()

                    # create a df with only covered to replace the original
                    covered_df = stats_df[~stats_df['#SAMPLE'].isin(
                        uncovered_samples)]
                    covered_df.to_csv(coverage_stat_file,
                                      sep="\t", index=False)
                    # create a df with uncovered
                    uncovered_df = stats_df[stats_df['#SAMPLE'].isin(
                        uncovered_samples)]
                    uncovered_table_filename = right_now_full + '_uncovered.summary.tab'
                    uncovered_table_file = os.path.join(
                        uncovered_dir, uncovered_table_filename)
                    if len(uncovered_samples) > 0:
                        uncovered_df.to_csv(
                            uncovered_table_file, sep="\t", index=False)
                elif name.endswith('.coverage.summary.tab'):
                    covstats_df = pd.read_csv(filename, sep="\t")
                    final_covstat = filename

    uncovered_samples = [str(x) for x in uncovered_samples]
    def_covstats_df = covstats_df[~covstats_df['#SAMPLE'].isin(
        uncovered_samples)]
    def_covstats_df.to_csv(final_covstat, sep="\t", index=False)

    logger.debug("Uncovered_samples:")
    logger.debug(uncovered_samples)

    # Remove other files
    for root, _, files in os.walk(output_dir):
        for name in files:
            if name.endswith('.cov') or name.endswith('.bamstats'):
                filename = os.path.join(root, name)
                #sample = re.search(r'^(.+?)[.]', name).group(1)
                sample = name.split(".")[0]
                if sample in uncovered_samples:
                    logger.debug(
                        "Removing FAULTY file {}".format(filename))
                    os.remove(filename)

    sample_list_F = []

    r1, r2, tmp = extract_read_list(output_dir)
    for r1_file, r2_file in zip(r1, r2):
        sample = extract_sample(r1_file, r2_file)
        sample_list_F.append(sample)

    # MOVE Fastq
    if len(uncovered_samples) > 0:
        for uncovered_sample in uncovered_samples:
            try:
                uncovered_index = sample_list_F.index(uncovered_sample)
                destination_file_r1 = os.path.join(
                    uncovered_dir, r1[uncovered_index].split("/")[-1])
                destination_file_r2 = os.path.join(
                    uncovered_dir, r2[uncovered_index].split("/")[-1])
                logger.debug("Moving FAULTY fastas {} AND {} TO {} AND {}".format(
                    r1[uncovered_index], r2[uncovered_index], destination_file_r1, destination_file_r2))
                shutil.move(r1[uncovered_index], destination_file_r1)
                shutil.move(r2[uncovered_index], destination_file_r2)
            except:
                logger.info('ERROR: No uncovered detected')

    # Move Variant Folder

    for root, _, files in os.walk(output_dir):
        if root.endswith('Variants') and not "Uncovered" in root:
            if len(uncovered_samples) > 0:
                for uncovered in uncovered_samples:
                    filename = os.path.join(root, uncovered)
                    destination_file = os.path.join(
                        uncovered_dir_variants, uncovered)
                    if not os.path.exists(destination_file):
                        logger.debug("Moving FAULTY folder {} TO {}".format(
                            filename, destination_file))
                        shutil.move(filename, destination_file)
                    else:
                        logger.debug(
                            "{} already exist".format(destination_file))

    return uncovered_samples


def clean_unwanted_files(args):
    Trimmed_dir = ""
    for root, _, files in os.walk(args.output):
        if 'Trimmed' in root:
            Trimmed_dir = root
        for name in files:
            filename = os.path.join(root, name)
            if root.endswith("Bam") and not "bqsr" in filename:
                logger.info("Removed: " + filename)
                os.remove(filename)
            # elif filename.endswith("cohort.g.vcf") or filename.endswith("cohort.g.vcf.idx"):
            #    print("Removed: " + filename)
            #    os.remove(filename)
            elif root.endswith("Annotation") and (filename.endswith("annot.genes.txt") or filename.endswith(".vcf") or filename.endswith(".annot.html")):
                logger.info("Removed: " + filename)
                os.remove(filename)
            elif root.endswith("Trimmed"):
                logger.info("Removed: " + filename)
                os.remove(filename)

    if Trimmed_dir:
        logger.info("Removed folder: " + Trimmed_dir)
        os.rmdir(Trimmed_dir)


def longest_common_suffix(list_of_strings):
    """
    Return the longest common suffix in a list of strings
    Adapted from https://gist.github.com/willwest/ca5d050fdf15232a9e67
    """
    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]
    return lcs


def list_to_bed(input_list, output_dir, output_file_name, reference="CHROM"):
    """
    Turn a list into a bed file with start and end position having the same value
    """
    output_dir = os.path.abspath(output_dir)

    output_bed_file = output_file_name + ".bed"

    final_output_path = os.path.join(output_dir, output_bed_file)

    if len(input_list) == 0:
        input_list.append(0)

    with open(final_output_path, 'w+') as f:
        for position in input_list:
            line = ("\t").join(
                [reference, str(position), str(position)]) + "\n"
            f.write(line)


def count_lines(input_file):
    with open(input_file, 'r') as f:
        content = f.read()
        content_list = content.split('\n')
        while '' in content_list:
            content_list.remove('')
    return len(content_list)


def check_reanalysis(output_dir, samples_to_analyze):
    output_dir = os.path.abspath(output_dir)
    new_samples = []
    #group = output_dir.split("/")[-1]

    variant_dir = os.path.join(output_dir, "Variants")
    compare_dir = os.path.join(output_dir, "Compare")

    previous_files = [variant_dir, compare_dir]

    # check how many folders exist
    file_exist = sum([os.path.exists(x)
                      for x in previous_files])  # True = 1, False = 0

    # Handle reanalysis: First time; reanalysis o reanalysis with aditional samples
    if file_exist > 0:  # Already analysed

        previous_samples_list = os.listdir(variant_dir)

        if len(samples_to_analyze) == len(previous_samples_list):

            logger.info(
                MAGENTA + "\nPREVIOUS ANALYSIS DETECTED, NO NEW SEQUENCES ADDED\n" + END_FORMATTING)
        else:
            new_samples = set(samples_to_analyze) - set(previous_samples_list)
            logger.info(
                MAGENTA + "\nPREVIOUS ANALYSIS DETECTED, " + str(len(new_samples)) + " NEW SEQUENCES ADDED\n" + END_FORMATTING)

    return list(new_samples)


def extrach_variants_summary(vcf_table, distance=15, quality=10):
    sample = vcf_table.split("/")[-1].split(".")[0]

    df = pd.read_csv(vcf_table, sep="\t", header=0)

    total_snp = len(df[df.TYPE == "SNP"].index)
    total_indels = len(df[df.TYPE == "INDEL"].index)
    total_homozygous = len(df[(df.TYPE == "SNP") & (df.gt0 == 1)].index)
    total_heterozygous = len(df[(df.TYPE == "SNP") & (df.gt0 == 0)].index)
    median_allele_freq = "%.2f" % (df.AF[df.TYPE == "SNP"].median())
    mean_allele_freq = "%.2f" % (df.AF[df.TYPE == "SNP"].mean())

    distance = distance
    QD = quality
    position_to_filter = df['POS'][((df.snp_left_distance <= distance) |
                                    (df.snp_right_distance <= distance) |
                                    (df.window_10 >= 2) |
                                    (df.AF <= 0.0) |
                                    (df.len_AD > 2) |
                                    (df.TYPE != "SNP") |
                                    (df.QD <= QD) |
                                    (df.highly_hetz == True) |
                                    (df.poorly_covered == True) |
                                    (df.non_genotyped == True))].tolist()

    filtered_df = df[~df.POS.isin(position_to_filter)]

    filtered_df_htz = filtered_df[filtered_df.gt0 == 0]

    ftotal_snp = len(filtered_df[filtered_df.TYPE == "SNP"].index)
    ftotal_homozygous = len(
        filtered_df[(filtered_df.TYPE == "SNP") & (filtered_df.gt0 == 1)].index)
    ftotal_heterozygous = len(
        filtered_df[(filtered_df.TYPE == "SNP") & (filtered_df.gt0 == 0)].index)
    fmedian_allele_freq = "%.2f" % (
        filtered_df.AF[filtered_df.TYPE == "SNP"].median())
    fmean_allele_freq = "%.2f" % (
        filtered_df.AF[filtered_df.TYPE == "SNP"].mean())
    fmean_allele_freq_htz = "%.2f" % (
        filtered_df_htz.AF[filtered_df_htz.TYPE == "SNP"].mean())

    output = [sample,
              total_snp,
              total_indels,
              total_homozygous,
              total_heterozygous,
              median_allele_freq,
              mean_allele_freq,
              ftotal_snp,
              ftotal_homozygous,
              ftotal_heterozygous,
              fmedian_allele_freq,
              fmean_allele_freq,
              fmean_allele_freq_htz]
    output = [str(x) for x in output]

    return "\t".join(output)


def vcf_stats(folder_table, distance=15, quality=10):

    out_file = os.path.join(folder_table, "vcf_stat.tab")
    mixed_samples = []

    with open(out_file, 'w+') as fout:
        fout.write("\t".join(["SAMPLE",
                              "#SNP",
                              "#INDELS",
                              "#HOMOZ_SNP",
                              "#HETZ_SNP",
                              "MEDIAN_AF_SNP",
                              "MEAN_AF_SNP",
                              "#FSNP",
                              "#FHOMOZ_SNP",
                              "#FHETZ_SNP",
                              "FMEDIAN_AF_SNP",
                              "FMEAN_AF_SNP",
                              "FMEAN_AF_SNP_HTZ"]))
        fout.write("\n")
        for root, _, files in os.walk(folder_table):
            for name in files:
                filename = os.path.join(root, name)
                if filename.endswith("raw.tab"):
                    line = extrach_variants_summary(filename)
                    line_split = line.split("\t")
                    sample = line_split[0]
                    htz_filtered = line_split[9]
                    if int(htz_filtered) > 100:
                        mixed_samples.append(sample)
                    fout.write(line)
                    fout.write("\n")
    return mixed_samples
