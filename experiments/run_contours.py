# TEST SCRIPT TO RUN CODE WITHOUT DEBUG SLOWDOWN

import gc
import logging
import pathlib
from cProfile import Profile
from datetime import datetime
from pstats import SortKey, Stats

import numpy as np

from pyalgcon.pipelines.generate_algebraic_contours import \
    generate_algebraic_contours

current_dir = pathlib.Path(__file__).parent
cam_file = current_dir / "data" / "cameras" / "camera_matrix_identity.csv"
obj_file = current_dir / "data" / "meshes" / "spot_quadrangulated_tri_clean_conf_simplified_with_uv.obj"
# obj_file = current_dir / "data" / "meshes" / "FAIL_pawn_tri_clean_conf_simplified_with_uv.obj"
# obj_file = current_dir / "data" / "meshes" / "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
out_file = current_dir / "contours_nu.svg"

camera_matrix = np.loadtxt(cam_file, delimiter=",")
print(camera_matrix)

# TODO: see if this changes anything
# gc.disable()
# gc.enable()
# gc.set_threshold(50000, 10, 10)

logging.disable(logging.CRITICAL)  # suppresses all logging calls below CRITICAL
# generate_algebraic_contours(camera_matrix, obj, obj.parent / "contours_nu.svg")

# OLD PYAC
# with Profile() as prof:
#     generate_algebraic_contours(camera_matrix, obj)
#     timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%SZ")
#     with open(f"{timestamp}-cumtime - OLD PYAC.log", "w", encoding="utf-8") as stream:
#         stats = (
#             Stats(prof, stream=stream)
#             .strip_dirs()
#             .sort_stats(SortKey.CUMULATIVE)
#             .print_stats()
#         )

#     with open(f"{timestamp}-tottime - OLD PYAC.log", "w", encoding="utf-8") as stream:
#         stats = (
#             Stats(prof, stream=stream)
#             .strip_dirs()
#             .sort_stats(SortKey.TIME)
#             .print_stats()
#         )

generate_algebraic_contours(camera_matrix, obj_file,  out_file)


# # NEW PYAC
# with Profile() as prof:
#     generate_algebraic_contours(camera_matrix, obj_file,  out_file)
#     timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%SZ")

#     with open(f"{timestamp}-cumtime - NEW PYAC.log", "w", encoding="utf-8") as stream:
#         stats = (
#             Stats(prof, stream=stream)
#             .strip_dirs()
#             .sort_stats(SortKey.CUMULATIVE)
#             .print_stats()
#         )

#     with open(f"{timestamp}-tottime - NEW PYAC.log", "w", encoding="utf-8") as stream:
#         stats = (
#             Stats(prof, stream=stream)
#             .strip_dirs()
#             .sort_stats(SortKey.TIME)
#             .print_stats()
#         )

# # with open(f"{timestamp}-percall.log", "w", encoding="utf-8") as stream:
# #     stats = (
# #         Stats(prof, stream=stream)
# #         .strip_dirs()
# #         .sort_stats(SortKey.PCALLS)
# #         .print_stats()
# #     )

# # with open(f"{timestamp}-calls.log", "w", encoding="utf-8") as stream:
# #     stats = (
# #         Stats(prof, stream=stream)
# #         .strip_dirs()
# #         .sort_stats(SortKey.CALLS)
# #         .print_stats()
# #         .print_callers('numpy.array')
# #     )

# with open(f"{timestamp}-nparray.log", "w", encoding="utf-8") as stream:
#     stats = (
#         Stats(prof, stream=stream)
#         .strip_dirs()
#         .sort_stats(SortKey.TIME)
#         .print_callers('numpy.array')
#     )
