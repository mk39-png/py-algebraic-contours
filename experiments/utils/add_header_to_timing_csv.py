#!/usr/bin/env python3

# Helper executable to add a header to the timing csv file

# This is a separate script from generate_timing_data.py since generate_timing_data.py may
# be used to call multiple meshes, of which it would create multiple unnecessary headers
# for a singular .csv file

import argparse
import pathlib
import sys


def main(args):
    """
    Method to add a header to the timing .csv file, overwriting the pre-existing files
    with the same names.
    """
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)

    # Write view independent timing data header
    with open(output_dir / "view_independent.csv", "w", encoding="utf-8") as out_view_independent:
        out_view_independent.write("mesh name, camera filename, num triangles, time spline surface,"
                                   " time patch boundary edges\n")

    # Write view dependent timing data header
    with open(output_dir / "per_view.csv", "w", encoding="utf-8") as out_per_view:
        out_per_view.write("mesh name, camera filename, rotation frame, z distance, "
                           "transformation time, "
                           "total time per view, surface update, "
                           "compute contour, compute cusps, compute intersections, compute "
                           "visibility, graph building, num segments, num int cusps, num bound "
                           "cusps, num intersection call, num ray inter call, num patches\n")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="add_header_to_timing_csv",
        description="Adds header to timing data CSVs.")
    parser.add_argument("output_dir",
                        nargs="?",
                        default=".",
                        type=str,
                        help="Output directory")
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
