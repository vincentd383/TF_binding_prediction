from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

PWM_PATH = RAW_DIR / "factorbookMotifPwm.txt.gz"

POSITIVE_PATH = (
    PROCESSED_DIR
    / "ctcf_positive_sites_with_clusters.tsv.gz"
)

SCORED_OUTPUT_PATH = (
    PROCESSED_DIR
    / "ctcf_positive_sites_pwm_scored.tsv.gz"
)

REPRESENTATIVE_OUTPUT_PATH = (
    PROCESSED_DIR
    / "ctcf_positive_pwm_representatives.tsv.gz"
)

SUMMARY_PATH = (
    PROCESSED_DIR
    / "ctcf_pwm_validation_summary.txt"
)

TF_NAME = "CTCF"
EXPECTED_LENGTH = 15
PSEUDOCOUNT = 1e-6

BASES = ("A", "C", "G", "T")
BASE_TO_INDEX = {
    base: index
    for index, base in enumerate(BASES)
}

BACKGROUND = np.asarray(
    [0.25, 0.25, 0.25, 0.25],
    dtype=float,
)

COMPLEMENT_TABLE = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
    }
)


# ---------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------

def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_TABLE)[::-1]


# ---------------------------------------------------------
# PWM parsing
# ---------------------------------------------------------

def parse_array(text: str) -> np.ndarray:
    values = [
        float(value)
        for value in text.split(",")
        if value.strip()
    ]

    return np.asarray(values, dtype=float)


def load_ctcf_pwm() -> np.ndarray:
    pwm_table = pd.read_csv(
        PWM_PATH,
        sep="\t",
        header=None,
        names=["tf", "length", "A", "C", "G", "T"],
        compression="gzip",
    )

    matching_rows = pwm_table[
        pwm_table["tf"].eq(TF_NAME)
    ]

    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one {TF_NAME} PWM, "
            f"found {len(matching_rows)}."
        )

    row = matching_rows.iloc[0]

    motif_length = int(row["length"])

    if motif_length != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected a {EXPECTED_LENGTH}-bp PWM, "
            f"found {motif_length} bp."
        )

    probability_matrix = np.vstack(
        [
            parse_array(row["A"]),
            parse_array(row["C"]),
            parse_array(row["G"]),
            parse_array(row["T"]),
        ]
    )

    if probability_matrix.shape != (4, EXPECTED_LENGTH):
        raise ValueError(
            f"Unexpected PWM shape: {probability_matrix.shape}"
        )

    original_column_sums = probability_matrix.sum(axis=0)

    if not np.allclose(
        original_column_sums,
        1.0,
        atol=0.02,
    ):
        raise ValueError(
            "The original probability columns do not "
            "sum approximately to one."
        )

    # Add a small pseudocount to prevent log2(0).
    adjusted_matrix = probability_matrix + PSEUDOCOUNT

    adjusted_matrix = (
        adjusted_matrix
        / adjusted_matrix.sum(axis=0, keepdims=True)
    )

    return probability_matrix, adjusted_matrix


def probability_to_log_odds(
    probability_matrix: np.ndarray,
) -> np.ndarray:

    return np.log2(
        probability_matrix
        / BACKGROUND[:, np.newaxis]
    )


# ---------------------------------------------------------
# PWM scoring
# ---------------------------------------------------------

def score_sequence(
    sequence: str,
    log_odds_matrix: np.ndarray,
) -> float:

    if len(sequence) != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_LENGTH} bases, "
            f"found {len(sequence)}."
        )

    score = 0.0

    for position, base in enumerate(sequence):
        try:
            base_index = BASE_TO_INDEX[base]
        except KeyError as error:
            raise ValueError(
                f"Unexpected base {base!r} in {sequence!r}."
            ) from error

        score += log_odds_matrix[
            base_index,
            position,
        ]

    return score


def theoretical_score_limits(
    log_odds_matrix: np.ndarray,
) -> tuple[float, float]:

    minimum_score = float(
        log_odds_matrix.min(axis=0).sum()
    )

    maximum_score = float(
        log_odds_matrix.max(axis=0).sum()
    )

    return minimum_score, maximum_score


# ---------------------------------------------------------
# Information-content calculation
# ---------------------------------------------------------

def calculate_information_content(
    probability_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    information_by_position = np.zeros(
        probability_matrix.shape[1],
        dtype=float,
    )

    contributions = np.zeros_like(
        probability_matrix,
        dtype=float,
    )

    for position in range(
        probability_matrix.shape[1]
    ):
        probabilities = probability_matrix[:, position]

        entropy = -np.sum(
            probabilities
            * np.log2(probabilities)
        )

        information = 2.0 - entropy

        information_by_position[position] = information

        contributions[:, position] = (
            probabilities * information
        )

    return information_by_position, contributions


# ---------------------------------------------------------
# Figures
# ---------------------------------------------------------

def plot_pwm_probabilities(
    probability_matrix: np.ndarray,
) -> None:

    positions = np.arange(
        1,
        EXPECTED_LENGTH + 1,
    )

    fig, axis = plt.subplots(figsize=(11, 5))

    bottom = np.zeros(EXPECTED_LENGTH)

    colors = {
        "A": "#2ca02c",
        "C": "#1f77b4",
        "G": "#ff7f0e",
        "T": "#d62728",
    }

    for base_index, base in enumerate(BASES):
        values = probability_matrix[base_index]

        axis.bar(
            positions,
            values,
            bottom=bottom,
            label=base,
            color=colors[base],
        )

        bottom += values

    axis.set_xlabel("Motif position")
    axis.set_ylabel("Nucleotide probability")
    axis.set_title("Supplied CTCF position weight matrix")
    axis.set_xticks(positions)
    axis.set_ylim(0, 1)
    axis.legend(title="Base", ncol=4)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ctcf_pwm_probabilities.png",
        dpi=300,
    )

    plt.close(fig)


def plot_information_content(
    information_by_position: np.ndarray,
) -> None:

    positions = np.arange(
        1,
        EXPECTED_LENGTH + 1,
    )

    fig, axis = plt.subplots(figsize=(10, 4))

    axis.bar(
        positions,
        information_by_position,
        color="#4c78a8",
    )

    axis.set_xlabel("Motif position")
    axis.set_ylabel("Information content (bits)")
    axis.set_title("CTCF PWM information content")
    axis.set_xticks(positions)
    axis.set_ylim(0, 2)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ctcf_pwm_information_content.png",
        dpi=300,
    )

    plt.close(fig)


def plot_score_distributions(
    sites: pd.DataFrame,
) -> None:

    fig, axis = plt.subplots(figsize=(9, 5))

    axis.hist(
        sites["oriented_pwm_score"],
        bins=60,
        alpha=0.65,
        label="Supplied orientation",
    )

    axis.hist(
        sites["reverse_pwm_score"],
        bins=60,
        alpha=0.45,
        label="Opposite orientation",
    )

    axis.set_xlabel("PWM log-odds score")
    axis.set_ylabel("Number of positive sites")
    axis.set_title(
        "CTCF PWM scores in the two orientations"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ctcf_pwm_score_distributions.png",
        dpi=300,
    )

    plt.close(fig)


def plot_score_relationship(
    sites: pd.DataFrame,
) -> None:

    sample_size = min(20_000, len(sites))

    sample = sites.sample(
        n=sample_size,
        random_state=561,
    )

    fig, axis = plt.subplots(figsize=(6, 6))

    axis.scatter(
        sample["supplied_score"],
        sample["oriented_pwm_score"],
        s=6,
        alpha=0.2,
    )

    axis.set_xlabel("Supplied Factorbook score")
    axis.set_ylabel("Calculated PWM log-odds score")
    axis.set_title(
        "Supplied score versus calculated CTCF PWM score"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ctcf_supplied_vs_calculated_score.png",
        dpi=300,
    )

    plt.close(fig)


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CTCF PWM...")

    (
        original_probability_matrix,
        adjusted_probability_matrix,
    ) = load_ctcf_pwm()

    log_odds_matrix = probability_to_log_odds(
        adjusted_probability_matrix
    )

    minimum_score, maximum_score = (
        theoretical_score_limits(log_odds_matrix)
    )

    information_by_position, _ = (
        calculate_information_content(
            adjusted_probability_matrix
        )
    )

    consensus_sequence = "".join(
        BASES[index]
        for index in np.argmax(
            adjusted_probability_matrix,
            axis=0,
        )
    )

    print("Loading positive sites...")

    sites = pd.read_csv(
        POSITIVE_PATH,
        sep="\t",
        compression="gzip",
    )

    if not sites["core_sequence"].str.len().eq(
        EXPECTED_LENGTH
    ).all():
        raise ValueError(
            "Positive sites contain unexpected "
            "core-sequence lengths."
        )

    print("Calculating PWM scores...")

    sites["oriented_pwm_score"] = [
        score_sequence(
            sequence,
            log_odds_matrix,
        )
        for sequence in sites["core_sequence"]
    ]

    sites["reverse_core_sequence"] = (
        sites["core_sequence"]
        .map(reverse_complement)
    )

    sites["reverse_pwm_score"] = [
        score_sequence(
            sequence,
            log_odds_matrix,
        )
        for sequence in sites["reverse_core_sequence"]
    ]

    sites["best_pwm_score"] = sites[
        [
            "oriented_pwm_score",
            "reverse_pwm_score",
        ]
    ].max(axis=1)

    sites["orientation_score_difference"] = (
        sites["oriented_pwm_score"]
        - sites["reverse_pwm_score"]
    )

    sites["supplied_orientation_is_best"] = (
        sites["oriented_pwm_score"]
        >= sites["reverse_pwm_score"]
    )

    sites["best_pwm_orientation"] = np.where(
        sites["supplied_orientation_is_best"],
        "supplied",
        "reverse",
    )

    correlation, correlation_pvalue = spearmanr(
        sites["supplied_score"],
        sites["oriented_pwm_score"],
    )

    orientation_agreement = (
        sites["supplied_orientation_is_best"].mean()
    )

    # Choose one final representative per overlap cluster,
    # based on our calculated score rather than supplied score.
    representatives = (
        sites.sort_values(
            [
                "cluster_id",
                "best_pwm_score",
                "supplied_score",
            ],
            ascending=[True, False, False],
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

    sites.to_csv(
        SCORED_OUTPUT_PATH,
        sep="\t",
        index=False,
        compression="gzip",
    )

    representatives.to_csv(
        REPRESENTATIVE_OUTPUT_PATH,
        sep="\t",
        index=False,
        compression="gzip",
    )

    plot_pwm_probabilities(
        original_probability_matrix
    )

    plot_information_content(
        information_by_position
    )

    plot_score_distributions(sites)
    plot_score_relationship(sites)

    summary_lines = [
        f"TF: {TF_NAME}",
        f"PWM length: {EXPECTED_LENGTH}",
        f"Consensus sequence: {consensus_sequence}",
        f"Positive motif records: {len(sites):,}",
        (
            "Overlap-cluster representatives: "
            f"{len(representatives):,}"
        ),
        (
            "Theoretical minimum score: "
            f"{minimum_score:.4f}"
        ),
        (
            "Theoretical maximum score: "
            f"{maximum_score:.4f}"
        ),
        (
            "Mean total information content: "
            f"{information_by_position.sum():.4f} bits"
        ),
        (
            "Supplied-orientation agreement: "
            f"{orientation_agreement:.3%}"
        ),
        (
            "Spearman correlation with supplied score: "
            f"{correlation:.4f}"
        ),
        (
            "Spearman correlation p-value: "
            f"{correlation_pvalue:.6g}"
        ),
        "",
        "Oriented PWM-score summary:",
        sites["oriented_pwm_score"].describe().to_string(),
        "",
        "Reverse PWM-score summary:",
        sites["reverse_pwm_score"].describe().to_string(),
        "",
        "Orientation-score difference summary:",
        (
            sites["orientation_score_difference"]
            .describe()
            .to_string()
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n"
    )

    print()
    print("\n".join(summary_lines))

    print(f"\nSaved scored sites to:\n{SCORED_OUTPUT_PATH}")
    print(
        "\nSaved final representatives to:\n"
        f"{REPRESENTATIVE_OUTPUT_PATH}"
    )
    print(f"\nSaved summary to:\n{SUMMARY_PATH}")
    print(f"\nSaved figures to:\n{FIGURE_DIR}")


if __name__ == "__main__":
    main()