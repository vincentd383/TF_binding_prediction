from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "ctcf_positive_sites.tsv.gz"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "ctcf_redundancy_summary.txt"
)

CLUSTER_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "ctcf_positive_sites_with_clusters.tsv.gz"
)

REPRESENTATIVE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "ctcf_positive_locus_representatives.tsv.gz"
)


def assign_overlap_clusters(
    chromosome_sites: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign sites to clusters of overlapping genomic intervals.

    Two sites belong to the same cluster if they overlap directly
    or are connected through a chain of overlapping sites.
    """

    chromosome_sites = chromosome_sites.sort_values(
        ["start", "end", "supplied_score"],
        ascending=[True, True, False],
    ).copy()

    cluster_numbers = []

    current_cluster = -1
    current_cluster_end = -1

    for site in chromosome_sites.itertuples():
        if site.start >= current_cluster_end:
            current_cluster += 1
            current_cluster_end = site.end
        else:
            current_cluster_end = max(
                current_cluster_end,
                site.end,
            )

        cluster_numbers.append(current_cluster)

    chromosome_sites["chromosome_cluster"] = cluster_numbers

    chromosome_sites["cluster_id"] = (
        chromosome_sites["chromosome"]
        + ":cluster_"
        + chromosome_sites["chromosome_cluster"].astype(str)
    )

    return chromosome_sites


def main() -> None:
    print("Loading CTCF positive sites...")

    sites = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        compression="gzip",
    )

    print(f"Positive records: {len(sites):,}")

    clustered_parts = []

    for chromosome, chromosome_sites in sites.groupby(
        "chromosome",
        sort=False,
    ):
        clustered = assign_overlap_clusters(
            chromosome_sites
        )

        clustered_parts.append(clustered)

        print(
            f"{chromosome}: "
            f"{len(chromosome_sites):,} sites, "
            f"{clustered['cluster_id'].nunique():,} "
            "overlap clusters"
        )

    clustered_sites = pd.concat(
        clustered_parts,
        ignore_index=True,
    )

    cluster_sizes = (
        clustered_sites.groupby("cluster_id")
        .size()
        .rename("cluster_size")
    )

    clustered_sites = clustered_sites.merge(
        cluster_sizes,
        left_on="cluster_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    # Count exact sequence duplicates separately.
    core_sequence_counts = (
        clustered_sites["core_sequence"].value_counts()
    )

    window_sequence_counts = (
        clustered_sites["window_sequence"].value_counts()
    )

    # Select one representative from every overlap cluster:
    # the site with the highest supplied score.
    representatives = (
        clustered_sites.sort_values(
            [
                "cluster_id",
                "supplied_score",
                "start",
            ],
            ascending=[True, False, True],
        )
        .drop_duplicates(
            subset=["cluster_id"],
            keep="first",
        )
        .sort_values(
            ["chromosome", "start", "end"]
        )
        .reset_index(drop=True)
    )

    summary = []

    summary.append(
        f"Total motif records: {len(clustered_sites):,}"
    )

    summary.append(
        "Unique genomic intervals: "
        f"{clustered_sites[['chromosome', 'start', 'end']].drop_duplicates().shape,}"
    )

    summary.append(
        "Unique core sequences: "
        f"{clustered_sites['core_sequence'].nunique():,}"
    )

    summary.append(
        "Unique 35-bp window sequences: "
        f"{clustered_sites['window_sequence'].nunique():,}"
    )

    summary.append(
        "Overlap clusters: "
        f"{clustered_sites['cluster_id'].nunique():,}"
    )

    summary.append(
        "Sites belonging to multi-site clusters: "
        f"{clustered_sites['cluster_size'].gt(1).sum():,}"
    )

    summary.append(
        "Fraction of sites in multi-site clusters: "
        f"{clustered_sites['cluster_size'].gt(1).mean():.3%}"
    )

    summary.append("")
    summary.append("Overlap-cluster size distribution:")
    summary.append(
        cluster_sizes.describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        ).to_string()
    )

    summary.append("")
    summary.append("Most frequent core-sequence counts:")
    summary.append(
        core_sequence_counts.head(20).to_string()
    )

    summary.append("")
    summary.append("Most frequent window-sequence counts:")
    summary.append(
        window_sequence_counts.head(20).to_string()
    )

    summary.append("")
    summary.append(
        "Representative sites after retaining one "
        f"per overlap cluster: {len(representatives):,}"
    )

    clustered_sites.to_csv(
        CLUSTER_PATH,
        sep="\t",
        index=False,
        compression="gzip",
    )

    representatives.to_csv(
        REPRESENTATIVE_PATH,
        sep="\t",
        index=False,
        compression="gzip",
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n"
    )

    print()
    print("\n".join(summary))

    print(f"\nSaved clustered sites to:\n{CLUSTER_PATH}")
    print(
        "\nSaved provisional representatives to:\n"
        f"{REPRESENTATIVE_PATH}"
    )
    print(f"\nSaved summary to:\n{SUMMARY_PATH}")


if __name__ == "__main__":
    main()