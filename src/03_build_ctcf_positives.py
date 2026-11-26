from pathlib import Path

import numpy as np
import pandas as pd
import pysam


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "datasets" / "reference" / "hg19"

MOTIF_POS_PATH = RAW_DIR / "factorbookMotifPos.txt.gz"

REGULATORY_PATH = (
    RAW_DIR
    / "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz"
)

REFERENCE_PATH = REFERENCE_DIR / "hg19.primary.fa"

OUTPUT_PATH = PROCESSED_DIR / "ctcf_positive_sites.tsv.gz"
SUMMARY_PATH = PROCESSED_DIR / "ctcf_positive_summary.txt"

TF_NAME = "CTCF"
MOTIF_LENGTH = 15
FLANK_LENGTH = 10
WINDOW_LENGTH = MOTIF_LENGTH + (2 * FLANK_LENGTH)

PRIMARY_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


# ---------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------

COMPLEMENT_TABLE = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
    }
)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    return sequence.upper().translate(COMPLEMENT_TABLE)[::-1]


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_binding_sites() -> pd.DataFrame:
    """Load the supplied Factorbook motif-position file."""

    column_names = [
        "record_id",
        "chromosome",
        "start",
        "end",
        "tf",
        "supplied_score",
        "strand",
    ]

    sites = pd.read_csv(
        MOTIF_POS_PATH,
        sep="\t",
        header=None,
        names=column_names,
        compression="gzip",
    )

    sites["site_length"] = sites["end"] - sites["start"]

    return sites


def load_regulatory_regions() -> pd.DataFrame:
    """Load the GM12878 active regulatory regions."""

    regions = pd.read_csv(
        REGULATORY_PATH,
        sep="\t",
        header=None,
        names=[
            "chromosome",
            "start",
            "end",
            "region_id",
        ],
        compression="gzip",
    )

    return regions


# ---------------------------------------------------------
# Active-region overlap
# ---------------------------------------------------------

def annotate_active_regions(
    sites: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annotate each site with the overlapping active regulatory region.

    The supplied regulatory-region file is merged, allowing a
    binary-search approach within each chromosome.
    """

    sites = sites.copy()
    sites["region_id"] = pd.NA

    for chromosome in PRIMARY_CHROMOSOMES:
        site_mask = sites["chromosome"].eq(chromosome)
        region_mask = regions["chromosome"].eq(chromosome)

        chromosome_sites = sites.loc[
            site_mask,
            ["start", "end"],
        ]

        chromosome_regions = (
            regions.loc[
                region_mask,
                ["start", "end", "region_id"],
            ]
            .sort_values(["start", "end"])
            .reset_index(drop=True)
        )

        if chromosome_sites.empty or chromosome_regions.empty:
            continue

        region_starts = chromosome_regions["start"].to_numpy()
        region_ends = chromosome_regions["end"].to_numpy()
        region_ids = chromosome_regions["region_id"].to_numpy()

        site_starts = chromosome_sites["start"].to_numpy()
        site_ends = chromosome_sites["end"].to_numpy()

        candidate_indices = (
            np.searchsorted(
                region_starts,
                site_ends,
                side="left",
            )
            - 1
        )

        valid = candidate_indices >= 0

        overlaps = np.zeros(
            len(chromosome_sites),
            dtype=bool,
        )

        overlaps[valid] = (
            region_ends[candidate_indices[valid]]
            > site_starts[valid]
        )

        overlapping_indices = chromosome_sites.index[overlaps]
        matching_region_indices = candidate_indices[overlaps]

        sites.loc[overlapping_indices, "region_id"] = (
            region_ids[matching_region_indices]
        )

    sites["overlaps_active_region"] = sites["region_id"].notna()

    return sites


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------

def remove_duplicate_intervals(
    sites: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Collapse records referring to the same genomic interval.

    If an interval appears more than once, retain the record with
    the highest supplied score. This also handles the uncommon case
    where the same coordinates appear with conflicting strands.
    """

    sites = sites.copy()

    exact_duplicate_count = sites.duplicated(
        subset=[
            "chromosome",
            "start",
            "end",
            "strand",
        ],
        keep=False,
    ).sum()

    coordinate_duplicate_count = sites.duplicated(
        subset=[
            "chromosome",
            "start",
            "end",
        ],
        keep=False,
    ).sum()

    sites = sites.sort_values(
        "supplied_score",
        ascending=False,
    )

    deduplicated = sites.drop_duplicates(
        subset=[
            "chromosome",
            "start",
            "end",
        ],
        keep="first",
    ).copy()

    deduplicated = deduplicated.sort_values(
        ["chromosome", "start", "end"]
    ).reset_index(drop=True)

    statistics = {
        "exact_duplicate_records": exact_duplicate_count,
        "coordinate_duplicate_records": coordinate_duplicate_count,
        "records_after_deduplication": len(deduplicated),
    }

    return deduplicated, statistics


# ---------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------

def extract_sequences(
    sites: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Extract core motifs and motif-plus-flank windows from hg19.

    pysam uses zero-based, half-open coordinates, matching BED.
    """

    extracted_records = []

    skipped_boundary = 0
    skipped_length = 0
    skipped_ambiguous = 0
    skipped_invalid_strand = 0

    with pysam.FastaFile(str(REFERENCE_PATH)) as genome:
        chromosome_lengths = {
            chromosome: genome.get_reference_length(chromosome)
            for chromosome in PRIMARY_CHROMOSOMES
        }

        for site in sites.itertuples(index=False):
            window_start = site.start - FLANK_LENGTH
            window_end = site.end + FLANK_LENGTH

            if (
                window_start < 0
                or window_end
                > chromosome_lengths[site.chromosome]
            ):
                skipped_boundary += 1
                continue

            core_sequence = genome.fetch(
                site.chromosome,
                site.start,
                site.end,
            ).upper()

            window_sequence = genome.fetch(
                site.chromosome,
                window_start,
                window_end,
            ).upper()

            if (
                len(core_sequence) != MOTIF_LENGTH
                or len(window_sequence) != WINDOW_LENGTH
            ):
                skipped_length += 1
                continue

            if site.strand == "-":
                core_sequence = reverse_complement(
                    core_sequence
                )
                window_sequence = reverse_complement(
                    window_sequence
                )
            elif site.strand != "+":
                skipped_invalid_strand += 1
                continue

            if (
                set(core_sequence) - {"A", "C", "G", "T"}
                or set(window_sequence) - {"A", "C", "G", "T"}
            ):
                skipped_ambiguous += 1
                continue

            extracted_records.append(
                {
                    "site_id": (
                        f"{site.chromosome}:"
                        f"{site.start}-{site.end}:"
                        f"{site.strand}"
                    ),
                    "chromosome": site.chromosome,
                    "start": site.start,
                    "end": site.end,
                    "strand": site.strand,
                    "tf": site.tf,
                    "supplied_score": site.supplied_score,
                    "region_id": site.region_id,
                    "site_length": site.site_length,
                    "window_start": window_start,
                    "window_end": window_end,
                    "core_sequence": core_sequence,
                    "window_sequence": window_sequence,
                    "gc_fraction": (
                        window_sequence.count("G")
                        + window_sequence.count("C")
                    )
                    / len(window_sequence),
                    "label": 1,
                }
            )

    extracted = pd.DataFrame(extracted_records)

    statistics = {
        "skipped_boundary": skipped_boundary,
        "skipped_wrong_sequence_length": skipped_length,
        "skipped_ambiguous_sequence": skipped_ambiguous,
        "skipped_invalid_strand": skipped_invalid_strand,
        "records_after_sequence_extraction": len(extracted),
    }

    return extracted, statistics


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_output(sites: pd.DataFrame) -> None:
    """Perform consistency checks before saving."""

    if sites.empty:
        raise ValueError("No valid CTCF sites were produced.")

    if not sites["site_length"].eq(MOTIF_LENGTH).all():
        raise ValueError("Not all CTCF sites have length 15.")

    if not sites["core_sequence"].str.len().eq(
        MOTIF_LENGTH
    ).all():
        raise ValueError("Incorrect core-sequence length detected.")

    if not sites["window_sequence"].str.len().eq(
        WINDOW_LENGTH
    ).all():
        raise ValueError("Incorrect window-sequence length detected.")

    if not sites["chromosome"].isin(
        PRIMARY_CHROMOSOMES
    ).all():
        raise ValueError("Non-autosomal records are present.")

    if not sites["strand"].isin(["+", "-"]).all():
        raise ValueError("Unexpected strand values are present.")

    valid_bases = {"A", "C", "G", "T"}

    invalid_sequence_count = sum(
        bool(set(sequence) - valid_bases)
        for sequence in sites["window_sequence"]
    )

    if invalid_sequence_count:
        raise ValueError(
            f"{invalid_sequence_count} sequences contain "
            "non-ACGT characters."
        )

    if sites["site_id"].duplicated().any():
        raise ValueError("Duplicate site identifiers remain.")

    if sites.duplicated(
        subset=["chromosome", "start", "end"]
    ).any():
        raise ValueError(
            "Duplicate genomic intervals remain."
        )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = []

    print("Loading binding sites...")
    all_sites = load_binding_sites()
    summary_lines.append(
        f"All supplied binding-site records: {len(all_sites):,}"
    )

    print("Selecting CTCF autosomal sites...")
    ctcf_sites = all_sites[
        all_sites["tf"].eq(TF_NAME)
        & all_sites["chromosome"].isin(
            PRIMARY_CHROMOSOMES
        )
    ].copy()

    summary_lines.append(
        f"Autosomal CTCF records: {len(ctcf_sites):,}"
    )

    ctcf_sites = ctcf_sites[
        ctcf_sites["site_length"].eq(MOTIF_LENGTH)
    ].copy()

    summary_lines.append(
        "Autosomal CTCF records with expected motif length: "
        f"{len(ctcf_sites):,}"
    )

    print("Loading active regulatory regions...")
    regions = load_regulatory_regions()

    regions = regions[
        regions["chromosome"].isin(
            PRIMARY_CHROMOSOMES
        )
    ].copy()

    print("Annotating active-region overlaps...")
    ctcf_sites = annotate_active_regions(
        ctcf_sites,
        regions,
    )

    ctcf_sites = ctcf_sites[
        ctcf_sites["overlaps_active_region"]
    ].copy()

    summary_lines.append(
        "CTCF records overlapping active regions: "
        f"{len(ctcf_sites):,}"
    )

    print("Removing duplicate intervals...")
    ctcf_sites, duplicate_statistics = (
        remove_duplicate_intervals(ctcf_sites)
    )

    for key, value in duplicate_statistics.items():
        summary_lines.append(f"{key}: {value:,}")

    print("Extracting hg19 sequences...")
    positive_sites, extraction_statistics = (
        extract_sequences(ctcf_sites)
    )

    for key, value in extraction_statistics.items():
        summary_lines.append(f"{key}: {value:,}")

    print("Validating final dataset...")
    validate_output(positive_sites)

    chromosome_counts = (
        positive_sites["chromosome"]
        .value_counts()
        .reindex(PRIMARY_CHROMOSOMES, fill_value=0)
    )

    strand_counts = positive_sites["strand"].value_counts()

    summary_lines.append("")
    summary_lines.append("Final chromosome counts:")

    for chromosome, count in chromosome_counts.items():
        summary_lines.append(
            f"  {chromosome}: {count:,}"
        )

    summary_lines.append("")
    summary_lines.append("Final strand counts:")

    for strand, count in strand_counts.items():
        summary_lines.append(f"  {strand}: {count:,}")

    summary_lines.append("")
    summary_lines.append(
        "Supplied-score summary:"
    )

    summary_lines.append(
        positive_sites["supplied_score"]
        .describe()
        .to_string()
    )

    summary_lines.append("")
    summary_lines.append(
        "Window GC-fraction summary:"
    )

    summary_lines.append(
        positive_sites["gc_fraction"]
        .describe()
        .to_string()
    )

    positive_sites.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        compression="gzip",
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n"
    )

    print("\n".join(summary_lines))

    print(f"\nSaved positive sites to:\n{OUTPUT_PATH}")
    print(f"\nSaved summary to:\n{SUMMARY_PATH}")

    print("\nFirst five final records:")
    print(
        positive_sites.head().to_string(index=False)
    )


if __name__ == "__main__":
    main()