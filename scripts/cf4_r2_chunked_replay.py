"""One full-target checkpoint replay, without an optimizer transition."""
from cf4_r2_conditional_map import BASE, main

if __name__ == '__main__':
    main(start_checkpoint=BASE / 'r2_conditional_full_gradient78_20261010/gradient.npz',
         max_evaluations=1,max_iterations=1,max_line_search=1,maxcor=1,
         app_seconds=3600,support_chunk_cells=16384)
