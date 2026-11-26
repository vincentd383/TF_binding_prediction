from pathlib import Path
import gzip

import pandas as pd
import pysam


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "datasets" / "reference" / "hg19"

MOTIF_POS_PATH = RAW_DIR / "factorbookMotifPos.txt.gz"
PWM_PATH = RAW_DIR / "factorbookMotifPwm.txt.gz"
REGULATORY_PATH = (
    RAW_DIR / "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz"
)
REFERENCE_PATH = REFERENCE_DIR / "hg19.primary.fa"


# We will initially restrict the project to the 22 autosomes.
PRIMARY_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def require_file(path: Path) -> None:
    """Raise a clear error if an expected file is absent."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")


def preview_gzip_file(path: Path, number_of_lines: int = 5) -> None:
    """Print the first few lines of a gzip-compressed text file."""

    print(f"\nFirst {number_of_lines} lines of {path.name}:")

    with gzip.open(path, mode="rt") as handle:
        for index, line in enumerate(handle):
            if index >= number_of_lines:
                break

            print(repr(line.rstrip("\n")))


def count_text_lines(path: Path) -> int:
    """Count lines without decompressing the file to disk."""

    with gzip.open(path, mode="rt") as handle:
        return sum(1 for _ in handle)


# ---------------------------------------------------------
# Reference-genome audit
# ---------------------------------------------------------

def audit_reference() -> set[str]:
    print("\n" + "=" * 70)
    print("REFERENCE GENOME")
    print("=" * 70)

    require_file(REFERENCE_PATH)
    require_file(Path(str(REFERENCE_PATH) + ".fai"))

    with pysam.FastaFile(str(REFERENCE_PATH)) as genome:
        references = list(genome.references)
        lengths = dict(zip(genome.references, genome.lengths))

        print(f"Reference file: {REFERENCE_PATH}")
        print(f"Number of reference sequences: {len(references)}")
        print(f"Reference names: {references}")

        missing_primary = [
            chromosome
            for chromosome in PRIMARY_CHROMOSOMES
            if chromosome not in references
        ]

        if missing_primary:
            raise ValueError(
                "Missing primary chromosomes: "
                + ", ".join(missing_primary)
            )

        print("\nPrimary chromosome lengths:")

        for chromosome in PRIMARY_CHROMOSOMES:
            print(f"  {chromosome}: {lengths,} bp")

        test_sequence = genome.fetch("chr1", 16245, 16260)

        print("\nReference-access test:")
        print(f"  Sequence: {test_sequence}")
        print(f"  Length: {len(test_sequence)}")

        if len(test_sequence) != 15:
            raise ValueError(
                "Reference fetch did not return the expected 15 bases."
            )

    return set(references)


# ---------------------------------------------------------
# Binding-site audit
# ---------------------------------------------------------

def audit_binding_sites(
    reference_chromosomes: set[str],
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("FACTORBOOK BINDING SITES")
    print("=" * 70)

    require_file(MOTIF_POS_PATH)
    preview_gzip_file(MOTIF_POS_PATH)

    column_names = [
        "record_id",
        "chromosome",
        "start",
        "end",
        "tf",
        "score",
        "strand",
    ]

    binding_sites = pd.read_csv(
        MOTIF_POS_PATH,
        sep=r"\s+",
        header=None,
        names=column_names,
        compression="gzip",
    )

    print(f"\nNumber of records: {len(binding_sites):,}")
    print(f"Number of columns: {binding_sites.shape[1]}")
    print(f"Missing values: {binding_sites.isna().sum().sum():,}")
    print(f"Number of TFs: {binding_sites['tf'].nunique():,}")

    print("\nData types:")
    print(binding_sites.dtypes)

    print("\nFirst five parsed rows:")
    print(binding_sites.head().to_string(index=False))

    print("\nMost frequent transcription factors:")
    tf_counts = binding_sites["tf"].value_counts()
    print(tf_counts.head(20).to_string())

    print("\nStrand values:")
    print(binding_sites["strand"].value_counts(dropna=False).to_string())

    print("\nChromosome values:")
    chromosome_counts = binding_sites["chromosome"].value_counts()
    print(chromosome_counts.head(30).to_string())

    invalid_coordinates = binding_sites[
        (binding_sites["start"] < 0)
        | (binding_sites["end"] <= binding_sites["start"])
    ]

    print(
        "\nRows with invalid coordinates: "
        f"{len(invalid_coordinates):,}"
    )

    binding_sites["site_length"] = (
        binding_sites["end"] - binding_sites["start"]
    )

    print("\nBinding-site length summary:")
    print(binding_sites["site_length"].describe().to_string())

    chromosomes_not_in_reference = sorted(
        set(binding_sites["chromosome"]) - reference_chromosomes
    )

    print("\nBinding-site chromosomes absent from reference:")
    print(
        chromosomes_not_in_reference
        if chromosomes_not_in_reference
        else "None"
    )

    autosomal_sites = binding_sites[
        binding_sites["chromosome"].isin(PRIMARY_CHROMOSOMES)
    ].copy()

    print(
        "\nRecords on chr1-chr22: "
        f"{len(autosomal_sites):,} "
        f"({len(autosomal_sites) / len(binding_sites):.2%})"
    )

    return binding_sites


# ---------------------------------------------------------
# Regulatory-region audit
# ---------------------------------------------------------

def audit_regulatory_regions(
    reference_chromosomes: set[str],
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("GM12878 ACTIVE REGULATORY REGIONS")
    print("=" * 70)

    require_file(REGULATORY_PATH)
    preview_gzip_file(REGULATORY_PATH)

    regulatory_regions = pd.read_csv(
        REGULATORY_PATH,
        sep="\t",
        header=None,
        compression="gzip",
        comment="#",
    )

    print(f"\nNumber of records: {len(regulatory_regions):,}")
    print(f"Number of columns: {regulatory_regions.shape[1]}")

    regulatory_regions = regulatory_regions.rename(
        columns={
            0: "chromosome",
            1: "start",
            2: "end",
        }
    )

    print("\nFirst five parsed rows:")
    print(regulatory_regions.head().to_string(index=False))

    required_columns = {"chromosome", "start", "end"}

    if not required_columns.issubset(regulatory_regions.columns):
        raise ValueError(
            "Regulatory-region file does not have at least "
            "three BED-style columns."
        )

    invalid_coordinates = regulatory_regions[
        (regulatory_regions["start"] < 0)
        | (
            regulatory_regions["end"]
            <= regulatory_regions["start"]
        )
    ]

    print(
        "\nRows with invalid coordinates: "
        f"{len(invalid_coordinates):,}"
    )

    regulatory_regions["region_length"] = (
        regulatory_regions["end"]
        - regulatory_regions["start"]
    )

    print("\nRegulatory-region length summary:")
    print(regulatory_regions["region_length"].describe().to_string())

    print("\nChromosome counts:")
    print(
        regulatory_regions["chromosome"]
        .value_counts()
        .head(30)
        .to_string()
    )

    chromosomes_not_in_reference = sorted(
        set(regulatory_regions["chromosome"])
        - reference_chromosomes
    )

    print("\nRegulatory chromosomes absent from reference:")
    print(
        chromosomes_not_in_reference
        if chromosomes_not_in_reference
        else "None"
    )

    return regulatory_regions


# ---------------------------------------------------------
# PWM file audit
# ---------------------------------------------------------

def audit_pwm_file() -> None:
    print("\n" + "=" * 70)
    print("FACTORBOOK PWM FILE")
    print("=" * 70)

    require_file(PWM_PATH)
    preview_gzip_file(PWM_PATH, number_of_lines=30)

    number_of_lines = count_text_lines(PWM_PATH)

    print(f"\nNumber of text lines: {number_of_lines:,}")

    header_candidates = []

    with gzip.open(PWM_PATH, mode="rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            if (
                stripped.startswith(">")
                or stripped.startswith("#")
                or stripped.startswith("MOTIF")
            ):
                header_candidates.append(
                    (line_number, stripped)
                )

            if len(header_candidates) >= 20:
                break

    print("\nPossible PWM header lines:")

    if header_candidates:
        for line_number, header in header_candidates:
            print(f"  Line {line_number}: {header}")
    else:
        print(
            "  No obvious headers detected. The file format "
            "must be inspected before writing the PWM parser."
        )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    print("COMP 561 DNA-shape project: initial data audit")
    print(f"Project root: {PROJECT_ROOT}")

    expected_files = [
        MOTIF_POS_PATH,
        PWM_PATH,
        REGULATORY_PATH,
        REFERENCE_PATH,
        Path(str(REFERENCE_PATH) + ".fai"),
    ]

    print("\nExpected-file check:")

    for path in expected_files:
        require_file(path)
        print(
            f"  OK: {path.relative_to(PROJECT_ROOT)} "
            f"({path.stat().st_size / 1_000_000:.2f} MB)"
        )

    reference_chromosomes = audit_reference()

    audit_binding_sites(reference_chromosomes)
    audit_regulatory_regions(reference_chromosomes)
    audit_pwm_file()

    print("\n" + "=" * 70)
    print("AUDIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()