#!/usr/bin/env python3

# Helper executable to add a header to the similarity csv file

import argparse
import pathlib
import sys


def main(args):
    """
    Method to add a header to the similarity .csv file, overwriting the pre-existing files
    with the same names.
    """
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)

    # Write view independent timing data header
    with open(output_dir / "similarity_metrics.csv", "w", encoding="utf-8") as out_similarity:
        out_similarity.write(
            "mesh name, camera filename, ASOC contours filename, PYAC contours filename, "
            "SSIM, MSE\n")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="add_header_to_similarity_csv",
        description="Adds header to similarity data CSVs.")
    parser.add_argument("output_dir",
                        nargs="?",
                        default=".",
                        type=str,
                        help="Output directory")
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
