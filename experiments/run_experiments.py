#!/usr/bin/env python

# NOTE: this was all ran assuming venv activate in Bash terminal using VSCode
# NOTE: as of right now, this script is primarily for Linux machines


import logging
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Any

#
# DEV NOTES
#

# Script that runs all of the Python scripts to compare with ASOC.
# Also, calls ASOC code itself as well.
# GOAL: things can be hardcoded in here for simplicity.

# NOTE: do NOT make a LOT of subdirectories of .csv files.
# NOTE: but subdirectories for pairs of PYAC and ASOC intermediate meshes are OK
# Instead, it should just be TWO CSVs, one for similarity and one for timing data.

# TODO: test each part one at a time!
# Which is to say, generate figures one at a time, starting with num_tests = 1.
# Then, branch outwards...
# b/c this might take a while
# TODO: also, utilize logging in case it fails at some point.

# -------------------------------
#
# EXPERIMENT SETUP
# 1. Establish directories to necessary folders.
# 2. Set up logging
# 3. Set up other variables needed for experiments
# 4. Call script to generate NUM_CAMERAS number of camera matrices
# 5. Get list of meshes and cameras for experimentation
# 6. Utility methods for validity and timing experiments
#
# -------------------------------

# -----------------
# 1. Directory setup
# -----------------

# NOTE: Directory relative to run_experiments.py script
experiments_dir: pathlib.Path = pathlib.Path(__file__).parent

# Data directories
meshes_dir: pathlib.Path = experiments_dir / "data" / "meshes"
cameras_dir: pathlib.Path = experiments_dir / "data" / "cameras"
results_dir: pathlib.Path = experiments_dir / "results"

# Make results directory if it does not exist.
cameras_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Script directories
metrics_scripts_dir: pathlib.Path = experiments_dir / "metrics/"
utils_scripts_dir: pathlib.Path = experiments_dir / "utils/"

# NOTE: Assuming that the algebraic contours directory has its default name and that
# it is in the same parent folder as PYAC.
# NOTE: this is assuming that the build for ASOC also exists as well
asoc_bin_dir: pathlib.Path = experiments_dir.parent.parent / "algebraic-contours" / "build" / "bin"
asoc_generate_timing_metrics: pathlib.Path = asoc_bin_dir / "generate_timing_metrics"
asoc_generate_similarity_metrics: pathlib.Path = asoc_bin_dir / "generate_similarity_metrics"

pyac_generate_timing_metrics: pathlib.Path = metrics_scripts_dir / "generate_timing_metrics.py"
pyac_generate_similiarity_metrics: pathlib.Path = metrics_scripts_dir / "generate_similarity_metrics.py"


# -----------------
# 2. Logging setup
# -----------------
logger: logging.Logger = logging.getLogger(__name__)
timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename: str = f"experiments_{timestamp}.log"
log_filepath: pathlib.Path = results_dir / "logs" / log_filename
log_filepath.parent.mkdir(parents=True, exist_ok=True)  # make log directory forcefully

logging.basicConfig(
    level=logging.NOTSET,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, mode="w"),  # Overwriting runs
    ]
)

# Logging directories
logger.info("experiments dir: %s", experiments_dir)
logger.info("meshes dir: %s", meshes_dir)
logger.info("camera dir: %s", cameras_dir)
logger.info("results dir: %s", results_dir)
logger.info("metrics scripts dur: %s", metrics_scripts_dir)
logger.info("utils scripts dir: %s", utils_scripts_dir)
logger.info("asoc bin dir: %s", asoc_bin_dir)
logger.info("asoc generate timing metrics dir: %s", asoc_generate_similarity_metrics)
logger.info("asoc generate similarity metrics dir: %s", asoc_generate_timing_metrics)

# -----------------
# 3. Other variables
# -----------------
NUM_CAMERA_MATRICES = 0

# -----------------
# 4. Generate camera matrices
# -----------------
subprocess.run([sys.executable, utils_scripts_dir / "generate_camera_matrices.py",
                "-o", str(cameras_dir),
                "--num_matrices", f"{NUM_CAMERA_MATRICES}"],
               check=True)

# -----------------
# 5. Get mesh and camera list
# -----------------
# NOTE: it may be more efficient to have all the camera matrices generated
#       in an application rather than reading it from disk, but saving to
#       .csv files on disk allows for us to experiment with them individually.
mesh_filenames: list[str] = [
    mesh_filepath.name
    for mesh_filepath in meshes_dir.iterdir()
    if mesh_filepath.suffix == ".obj"
]

camera_filenames: list[str] = [
    camera_filepath.name
    for camera_filepath in cameras_dir.iterdir()
    if camera_filepath.suffix == ".csv"
]
# Sort in order so we have consistent execution
camera_filenames.sort()

# log names
logger.info("mesh filenames: %s", mesh_filenames)
logger.info("camera filenames: %s", camera_filenames)

# -------------------------------
# 6. Utility methods
# -------------------------------


def run_script_with_logging(args: list[Any]) -> bool:
    """
    Utility wrapper that calls subprocess with logging to file.

    :param args: list of arguments for subprocess.run()
    :return: true upon success, otherwise false
    """
    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True)
        if proc.stdout:
            logger.info(proc.stdout.rstrip())
        if proc.stderr:
            logger.warning(proc.stderr.rstrip())

        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            "SCRIPT %s FAILED (return code %s)",
            args,
            e.returncode,
        )
        logger.error("STDOUT:\n%s", e.stdout)
        logger.error("STDERR:\n%s", e.stderr)

        return False


def run_timing_script(script_filepath: pathlib.Path,
                      output_dir: pathlib.Path,
                      _mesh_filenames: list[str],
                      _camera_filenames: list[str],
                      debug: bool = True) -> None:
    """
    Utility wrapper method that runs timing_data tests.
    Accepts script_filepath since needs to call ASOC or PYAC
    NOTE: this relies on ASOC and PYAC having the same parameter list.

    :param script_filepath: filepath to script executable
    :parma output_dir: path to output directory
    :param _mesh_filenames: list of mesh filename strings
    :param _camera_filenames: list of camera filename strings
    :param debug: determines if Python script runs with "assert" statements and logging.
                  Disable when measuring timing results.
    """
    # Tracker variables to see the progress made so far.
    success_counter: int = 0
    num_meshes: int = len(_mesh_filenames)
    num_cameras: int = len(_camera_filenames)

    # Run experiments themselves
    for mesh_filename in _mesh_filenames:
        for camera_filename in _camera_filenames:
            mesh_filepath: pathlib.Path = meshes_dir / mesh_filename
            camera_filepath: pathlib.Path = cameras_dir / camera_filename
            # output_mesh_camera_dir: pathlib.Path = output_dir / mesh_filename / camera_filename
            # output_mesh_camera_dir.mkdir(parents=True, exist_ok=True)

            # Call script with hange output directory depending on mesh_filepath
            # and camera_filepath...
            args: list[str] = [str(script_filepath),
                               "--input",  str(mesh_filepath),
                               "--output", str(output_dir),
                               "--camera", str(camera_filepath)]

            # If Python script, prepend sys.executable
            if script_filepath.suffix == ".py":
                if not debug:
                    args.insert(0, "-OO")  # prepend optimization flag first, removes asserts
                args.insert(0, sys.executable)  # prepend executable flag

            # Printing to console what script has been ran
            logger.info("\nrunning \n exec %s\n with mesh %s\n and camera %s\n to output %s",
                        script_filepath, mesh_filepath, camera_filepath, output_dir)
            logger.info("script ran: %s", " ".join(args))

            # Execute actual script itself
            if run_script_with_logging(args):
                success_counter += 1
            logger.info("%s out of %s meshes successfully ran using camera %s",
                        success_counter, num_meshes, camera_filepath)


def run_similarity_script(script_filepath: pathlib.Path,
                          output_dir: pathlib.Path,
                          _mesh_filenames: list[str],
                          _camera_filenames: list[str],
                          debug: bool = True) -> None:
    """
    Utility wrapper method that runs similarity metric experiments.
    Accepts script_filepath since needs to call ASOC or PYAC
    NOTE: this relies on ASOC and PYAC having the same parameter list.

    :param script_filepath: filepath to script executable
    :parma output_dir: path to output directory
    :param _mesh_filenames: list of mesh filename strings
    :param _camera_filenames: list of camera filename strings
    :param debug: determines if Python script runs with "assert" statements and logging. 
                  Disable when measuring timing results.
    """
    # Tracker variables to see the progress made so far.
    success_counter: int = 0
    num_meshes: int = len(_mesh_filenames)
    num_cameras: int = len(_camera_filenames)

    # Run experiments themselves
    for mesh_filename in _mesh_filenames:
        for camera_filename in _camera_filenames:
            mesh_filepath: pathlib.Path = meshes_dir / mesh_filename
            camera_filepath: pathlib.Path = cameras_dir / camera_filename

            # Like run_timing_script, but need to create subfolders for every mesh and its
            # associated camera matrix
            output_mesh_camera_dir: pathlib.Path = output_dir / mesh_filename / camera_filename
            output_mesh_camera_dir.mkdir(parents=True, exist_ok=True)

            # Call script with hange output directory depending on mesh_filepath
            # and camera_filepath...
            args: list[str] = [str(script_filepath),
                               "--input",  str(mesh_filepath),
                               "--output", str(output_mesh_camera_dir),
                               "--camera", str(camera_filepath)]

            # If Python script, prepend sys.executable
            if script_filepath.suffix == ".py":
                if not debug:
                    args.insert(0, "-OO")  # prepend optimization flag first, removes asserts
                args.insert(0, sys.executable)  # prepend executable flag

            # Printing to console what script has been ran
            logger.info("\nrunning \n exec %s\n with mesh %s\n and camera %s\n to output %s",
                        script_filepath, mesh_filepath, camera_filepath, output_mesh_camera_dir)
            logger.info("script ran: %s", " ".join(args))

            # Execute actual script itself
            if run_script_with_logging(args):
                success_counter += 1
            logger.info("%s out of %s meshes successfully ran using camera %s",
                        success_counter, num_meshes, camera_filepath)


def compare_similarity(_mesh_filenames: list[str],
                       _camera_filenames: list[str]) -> None:
    """
    Loop through all pairs of mesh and camera and performs SSIM metrics on the following:
    * contours.png
    * mesh.png
    * spline_surface.png
    """
    for mesh_filename in _mesh_filenames:
        for camera_filename in _camera_filenames:
            mesh_filepath: pathlib.Path = meshes_dir / mesh_filename
            camera_filepath: pathlib.Path = cameras_dir / camera_filename

            # Like run_timing_script, but need to create subfolders for every mesh and its
            # associated camera matrix
            output_mesh_camera_dir: pathlib.Path = results_dir / mesh_filename / camera_filename
            output_mesh_camera_dir.mkdir(parents=True, exist_ok=True)

            # TODO: load images
            # TODO: create plots of images and their calculated MSE and SSIM measurements
            # TODO: save to file


# -------------------------------
#
# ASOC VALIDITY EXPERIMENTS
#
# -------------------------------
# TODO: use generate_example_figure since that makes everything we need...
# And make directories and subdirectories depending on the mesh testing + ASOC vs PYAC...
results_asoc_dir: pathlib.Path = results_dir / "asoc/"
results_asoc_dir.mkdir(parents=True, exist_ok=True)
run_similarity_script(asoc_generate_similarity_metrics,
                      results_asoc_dir,
                      mesh_filenames,
                      camera_filenames)


# TODO: have script check if stuff runs OK or not....
results_pyac_dir: pathlib.Path = results_dir / "pyac/"
results_pyac_dir.mkdir(parents=True, exist_ok=True)
run_similarity_script(pyac_generate_similiarity_metrics,
                      results_pyac_dir,
                      mesh_filenames,
                      camera_filenames)


#
# Now, need to loop through ASOC and PYAC results and perform similarity metrics.
# Save such metrics into a .csv file.
#


# -------------------------------
#
# ASOC + PYAC TIMING EXPERIMENT
#
# -------------------------------

# Run ASOC timing metrics
results_asoc_dir: pathlib.Path = results_dir / "asoc/"
results_asoc_dir.mkdir(parents=True, exist_ok=True)
run_script_with_logging([sys.executable,
                         str(utils_scripts_dir / "add_header_to_similarity_csv.py"),
                         results_asoc_dir])
run_script_with_logging([sys.executable,
                         str(utils_scripts_dir / "add_header_to_timing_csv.py"),
                         results_asoc_dir])
# TODO: perhaps clear directories before starting to write into them...
run_timing_script(asoc_generate_timing_metrics,
                  results_asoc_dir,
                  mesh_filenames,
                  camera_filenames)

# Run PYAC timing metrics
results_pyac_dir: pathlib.Path = results_dir / "pyac/"
results_pyac_dir.mkdir(parents=True, exist_ok=True)
run_script_with_logging([sys.executable,
                         str(utils_scripts_dir / "add_header_to_similarity_csv.py"),
                         results_pyac_dir])
run_script_with_logging([sys.executable,
                         str(utils_scripts_dir / "add_header_to_timing_csv.py"),
                         results_pyac_dir])
run_timing_script(pyac_generate_timing_metrics,
                  results_pyac_dir,
                  mesh_filenames,
                  camera_filenames,
                  True)

exit(0)
