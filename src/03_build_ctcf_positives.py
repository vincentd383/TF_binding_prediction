from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths and project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

MOTIF_POS_PATH = RAW_DIR / "factorbookMotifPos.txt.gz"
PWM_PATH = RAW_DIR / "factorbookMotifPwm.txt.gz"
REGULATORY_PATH = (
    RAW_DIR / "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz"
)

OUTPUT_PATH = PROCESSED_DIR / "tf_selection_table.tsv"

PRIMARY_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


# ---------------------------------------------------------
# PWM parsing
# ---------------------------------------------------------

def parse_probability_array(text: str) -> np.ndarray:
    """
    Parse a comma-separated array from the PWM file.

    Each array has a trailing comma, so empty values are removed.
    """

    values = [
        float(value)
        for value in text.split(",")
        if value.strip() != ""
    ]

    return np.asarray(values, dtype=float)


def load_pwms(path: Path) -> dict:
    """
    Load the Factorbook PWM file.

    Expected columns:
        TF, length, A values, C values, G values, T values
    """

    pwm_table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["tf", "length", "A", "C", "G", "T"],
        compression="gzip",
    )

    print(f"PWM records read: {len(pwm_table):,}")

    if pwm_table["tf"].duplicated().any():
        duplicate_names = sorted(
            pwm_table.loc[
                pwm_table["tf"].duplicated(keep=False),
                "tf",
            ].unique()
        )

        raise ValueError(
            "Duplicate TF names found in PWM file: "
            + ", ".join(duplicate_names)
        )

    pwms = {}

    for row in pwm_table.itertuples(index=False):
        matrix = np.vstack(
            [
                parse_probability_array(row.A),
                parse_probability_array(row.C),
                parse_probability_array(row.G),
                parse_probability_array(row.T),
            ]
        )

        expected_length = int(row.length)

        if matrix.shape != (4, expected_length):
            raise ValueError(
                f"Invalid PWM shape for {row.tf}: "
                f"expected (4, {expected_length}), "
                f"found {matrix.shape}"
            )

        column_sums = matrix.sum(axis=0)

        if not np.allclose(
            column_sums,
            1.0,
            atol=0.02,
        ):
            maximum_error = np.max(np.abs(column_sums - 1.0))

            raise ValueError(
                f"PWM probabilities do not sum to one for "
                f"{row.tf}. Maximum error: {maximum_error:.6f}"
            )

        pwms[row.tf] = {
            "length": expected_length,
            "matrix": matrix,
        }

    print(f"Validated PWMs: {len(pwms):,}")

    return pwms


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_binding_sites() -> pd.DataFrame:
    column_names = [
        "record_id",
        "chromosome",
        "start",
        "end",
        "tf",
        "score",
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
# Interval overlap
# ---------------------------------------------------------

def mark_regulatory_overlaps(
    sites: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.Series:
    """
    Mark each site that overlaps at least one active regulatory region.

    The regulatory intervals are already merged, according to the
    supplied file name. For each chromosome, we use binary search
    against the sorted regulatory-region starts.

    BED overlap condition:
        site_start < region_end
        and site_end > region_start
    """

    overlaps = pd.Series(
        False,
        index=sites.index,
        dtype=bool,
    )

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
                ["start", "end"],
            ]
            .sort_values(["start", "end"])
            .reset_index(drop=True)
        )

        if chromosome_sites.empty or chromosome_regions.empty:
            continue

        region_starts = chromosome_regions["start"].to_numpy()
        region_ends = chromosome_regions["end"].to_numpy()

        site_starts = chromosome_sites["start"].to_numpy()
        site_ends = chromosome_sites["end"].to_numpy()

        # Find the rightmost regulatory interval whose start is
        # strictly smaller than the site end.
        candidate_indices = (
            np.searchsorted(
                region_starts,
                site_ends,
                side="left",
            )
            - 1
        )

        valid_candidates = candidate_indices >= 0

        chromosome_overlap = np.zeros(
            len(chromosome_sites),
            dtype=bool,
        )

        chromosome_overlap[valid_candidates] = (
            region_ends[candidate_indices[valid_candidates]]
            > site_starts[valid_candidates]
        )

        overlaps.loc[chromosome_sites.index] = chromosome_overlap

        print(
            f"{chromosome}: "
            f"{chromosome_overlap.sum():,} of "
            f"{len(chromosome_overlap):,} sites overlap "
            f"active regions"
        )

    return overlaps


# ---------------------------------------------------------
# Selection table
# ---------------------------------------------------------

def build_selection_table(
    sites: pd.DataFrame,
    pwms: dict,
) -> pd.DataFrame:

    pwm_lengths = {
        tf: information["length"]
        for tf, information in pwms.items()
    }

    sites["pwm_available"] = sites["tf"].isin(pwm_lengths)
    sites["pwm_length"] = sites["tf"].map(pwm_lengths)

    sites["length_matches_pwm"] = (
        sites["site_length"] == sites["pwm_length"]
    )

    records = []

    for tf, tf_sites in sites.groupby("tf", sort=True):
        autosomal = tf_sites[
            tf_sites["chromosome"].isin(PRIMARY_CHROMOSOMES)
        ]

        active = autosomal[
            autosomal["overlaps_active_region"]
        ]

        length_matched_active = active[
            active["length_matches_pwm"]
        ]

        records.append(
            {
                "tf": tf,
                "pwm_available": tf in pwms,
                "pwm_length": pwm_lengths.get(tf, np.nan),
                "total_sites": len(tf_sites),
                "autosomal_sites": len(autosomal),
                "active_region_sites": len(active),
                "length_matched_active_sites": len(
                    length_matched_active
                ),
                "active_chromosomes": (
                    length_matched_active["chromosome"].nunique()
                ),
                "positive_score_mean": (
                    length_matched_active["score"].mean()
                ),
                "positive_score_median": (
                    length_matched_active["score"].median()
                ),
                "positive_score_min": (
                    length_matched_active["score"].min()
                ),
                "positive_score_max": (
                    length_matched_active["score"].max()
                ),
            }
        )

    selection_table = pd.DataFrame(records)

    selection_table["active_fraction"] = (
        selection_table["active_region_sites"]
        / selection_table["autosomal_sites"]
    )

    selection_table["length_match_fraction"] = (
        selection_table["length_matched_active_sites"]
        / selection_table["active_region_sites"]
    )

    selection_table = selection_table.sort_values(
        [
            "length_matched_active_sites",
            "active_chromosomes",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return selection_table


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_overlap_examples(
    sites: pd.DataFrame,
    regions: pd.DataFrame,
) -> None:
    """
    Manually validate a small sample of positive overlap calls.
    """

    examples = sites[
        sites["overlaps_active_region"]
    ].sample(
        n=min(10, sites["overlaps_active_region"].sum()),
        random_state=561,
    )

    for site in examples.itertuples():
        matching_regions = regions[
            (regions["chromosome"] == site.chromosome)
            & (regions["start"] < site.end)
            & (regions["end"] > site.start)
        ]

        if matching_regions.empty:
            raise AssertionError(
                "Overlap-validation failure for "
                f"{site.chromosome}:{site.start}-{site.end}"
            )

    print("Random overlap validation passed.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and validating PWMs...")
    pwms = load_pwms(PWM_PATH)

    print("\nLoading binding sites...")
    sites = load_binding_sites()
    print(f"Binding sites loaded: {len(sites):,}")

    print("\nLoading active regulatory regions...")
    regions = load_regulatory_regions()
    print(f"Regulatory regions loaded: {len(regions):,}")

    print("\nRestricting to chromosomes 1-22...")
    sites = sites[
        sites["chromosome"].isin(PRIMARY_CHROMOSOMES)
    ].copy()

    regions = regions[
        regions["chromosome"].isin(PRIMARY_CHROMOSOMES)
    ].copy()

    print(f"Autosomal binding sites: {len(sites):,}")
    print(f"Autosomal regulatory regions: {len(regions):,}")

    print("\nCalculating overlaps...")
    sites["overlaps_active_region"] = (
        mark_regulatory_overlaps(sites, regions)
    )

    validate_overlap_examples(sites, regions)

    print("\nBuilding TF-selection table...")
    selection_table = build_selection_table(sites, pwms)

    selection_table.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print(f"\nSaved selection table to:\n{OUTPUT_PATH}")

    display_columns = [
        "tf",
        "pwm_available",
        "pwm_length",
        "total_sites",
        "active_region_sites",
        "length_matched_active_sites",
        "active_chromosomes",
        "active_fraction",
        "length_match_fraction",
    ]

    suitable = selection_table[
        selection_table["pwm_available"]
        & (
            selection_table["length_matched_active_sites"]
            >= 1000
        )
        & (
            selection_table["active_chromosomes"]
            >= 15
        )
    ]

    print("\nTop 25 transcription factors:")
    print(
        selection_table[display_columns]
        .head(25)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(
        "\nTFs meeting preliminary suitability criteria "
        "(at least 1,000 length-matched active sites and "
        "at least 15 chromosomes):"
    )

    if suitable.empty:
        print("None")
    else:
        print(
            suitable[display_columns]
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )


if __name__ == "__main__":
    main()