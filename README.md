# AutoSnippy: Whole-genome sequencing analysis pipeline for prokaryotes (specialized in _Mycobacterium tuberculosis_)

`AutoSnippy` is a modular, in-house pipeline designed for the analysis of paired-end WGS data from _Mycobacterium tuberculosis_. It performs quality control, variant calling, taxonomic assignment, variant annotation and comparative genomics in a reproducible and structured way.

---

## 📥 Clone the Repository

To get started, clone the `autosnippy` repository from GitHub:

```bash
git clone https://github.com/MG-IiSGM/autosnippy
cd autosnippy
```

---

## 🔧 Installation

This pipeline is built to run in a Conda environment. Use the provided `autosnippy.yml` file to install all dependencies:

```bash
conda env create -f autosnippy.yml
conda activate autosnippy
```

> Channels used: `bioconda`, `conda-forge`, `anaconda`, `r`, `defaults`, `etetoolkit`.

---

## 🚀 Usage

Run the pipeline from the command line using the main script:

```bash
python autosnippy.py -i input_folder -o output_folder -T 100 -r reference.fa --snpeff_database Mycobacterium_tuberculosis_h37rv --mash_database bacteria_mash.msh --kraken2 Kraken2_database/ -R repeats_phage_coord.bed -V lin_tbdb_MTBanc.vcf -V Resistances_OMS.vcf -A Resistance_OMS.aa
```

### 🔹 Parameters

| Parameter           | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `-i`                | Input folder with paired-end FASTQ files (`*_1.fastq.gz`, `*_2.fastq.gz`) |
| `-o`                | Output folder                                                             |
| `-T`                | Number of threads (optional)                                              |
| `-r`                | Reference genome in FASTA format                                          |
| `--snpeff_database` | SnpEff database name (optional, for variant annotation)                   |
| `--mash_database`   | Mash database for taxonomic classification (optional)                     |
| `--kraken2`         | Kraken2 database path (optional)                                          |
| `-R`                | BED file with coordinates to remove (e.g. repeats, phages) (optional)     |
| `-V`                | VCF file(s) with custom variant annotations (lineage/resistance markers)  |
| `-A`                | File with codon-level resistance annotations (requires SnpEff output)     |

---

### 📄 Example of Optional Input Files

The following optional inputs can be provided to enrich the variant annotation and filtering process:

#### 🔹 `-R` or `-B` : BED file with regions to exclude (`-R`) (e.g., repeats or phage insertions) or annotate any position within a region (`-B`)

This file defines genomic coordinates (e.g. in BED format) that will be excluded from downstream analysis, such as repetitive or poorly mappable regions in the _M. tuberculosis_ genome.

**Example:**

| Chromosome | Start  | End    | INFO |
| ---------- | ------ | ------ | ---- |
| MTB_anc    | 33582  | 33794  | +    |
| MTB_anc    | 80185  | 80373  | +    |
| MTB_anc    | 103710 | 104663 | -    |

---

#### 🔹 `-V` : VCF file with user-defined variant annotations (e.g., lineage or resistance markers)

This file should contain variants of interest to annotate VCF outputs during the analysis, such as lineage-defining mutations.

**Example:**

| CHROM      | POS  | ID  | REF | ALT | QUAL | FILTER | FORMAT | INFO           |
| ---------- | ---- | --- | --- | --- | ---- | ------ | ------ | -------------- |
| Chromosome | 1131 | .   | C   | A   | 20   | PASS   | .      | lineage4.2.2.1 |
| Chromosome | 4206 | .   | C   | T   | 20   | PASS   | .      | lineage4.1.3   |
| Chromosome | 9944 | .   | A   | C   | 20   | PASS   | .      | lineage4.2.2   |

---

#### 🔹 `-A` : Codon-level resistance annotations

This file allows you to specify amino acid substitutions linked to resistance phenotypes. These will be matched against variant calls (e.g. via SnpEff output).

**Example:**

| Substitution | Gene:Drug        |
| ------------ | ---------------- |
| Asp435Phe    | rpoB:Rifampicina |
| Asp435Tyr    | rpoB:Rifampicina |
| Asp435Val    | rpoB:Rifampicina |

> These inputs are optional, but if `-A` is specified, SnpEff annotation must be run (`--snpeff_database`) to enable codon matching.

---

## 📂 Output Structure

The pipeline produces a structured directory with the following subfolders:

- `Quality/` – FastQC reports for raw data.
- `Stats/`
  - `Coverage/` – per-sample `.cov` files for genome-wide depth.
  - `Bamstats/` – alignment stats in `.bamstats` format.
  - `coverage.summary.tab` – summary of coverage thresholds.
  - `overall.stats.tab` – includes high-quality SNPs, indels, heterozygosity and high-quality sequences.
- `Variants/` – one folder per sample with BAM, BAI, raw and filtered VCFs.
- `Species/` – taxonomic assignment results (if `--mash_database` or `--kraken2` used).
- `Uncovered/` – samples failing the 70% >20X coverage threshold (configurable).
- `Annotation/`
  - `snpeff/` – SnpEff-annotated VCFs.
  - `user/` – VCFs annotated with custom variant lists from `-V`.
  - `user_aa/` – codon-level resistance annotations from `-A`.
- `Compare/` – inter-sample SNV comparisons, phylogeny (`.nwk`), dendrograms and intermediate data.

> For ambiguous or low-confidence regions (e.g. heterozygous SNPs), manual review with a genome browser is recommended.

---

## ✅ Quality Control

- **FastQC** is used to assess raw data quality.
- **Filtering and trimming** are done via Snippy and custom wrappers.
- **Coverage and alignment statistics** are generated with `samtools` and integrated into summary reports.

---

## 📦 Reproducibility and Portability

We strongly recommend using containers for reproducibility. If you are using a shared cluster or HPC environment, you can build a Docker or Singularity image based on the `autosnippy.yml` environment file.

---

## 🧪 Example (Minimal)

```bash
python autosnippy.py -i . -o . -T 16 -r H37Rv.fasta
```

---

## 📬 Contact

For questions or contributions, please contact the authors or open an issue in the GitHub repository.

**Author:** [Sergio Buenestado-Serrano](mailto:sergio.buenestado@gmail.com)  
**Email:** <sergio.buenestado@gmail.com>

---

## 📄 License

This software is distributed under the **GNU GENERAL PUBLIC LICENSE**.

Although the project is distributed under an open license, **collaboration is highly welcome**. Contributions, feedback, and suggestions to improve AutoSnippy are encouraged.
