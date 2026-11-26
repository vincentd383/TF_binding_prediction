from pathlib import Path
import pysam

reference_path = Path(
    "datasets/reference/hg19/hg19.primary.fa"
)

genome = pysam.FastaFile(str(reference_path))

print(genome.references)
print(genome.get_reference_length("chr1"))

sequence = genome.fetch("chr1", 16245, 16260)

print(sequence)
print("Length:", len(sequence))

genome.close()