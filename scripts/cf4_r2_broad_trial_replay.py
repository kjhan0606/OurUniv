"""Replay recorded unaccepted trial 8; not the unsaved failing candidate."""
from cf4_r2_conditional_map import BASE, main

if __name__ == '__main__':
    main(recorded_trial=BASE / 'r2_joint_warm_start_20261010/best_parameters.npz',
         max_evaluations=1, max_iterations=1, max_line_search=1, maxcor=1,
         app_seconds=3600, support_chunk_cells=16384)
