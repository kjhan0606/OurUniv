"""One bounded, jointly scaled optimization of the unchanged conditional target."""
from cf4_r2_conditional_map import BASE, main


if __name__ == '__main__':
    # Fixed provisional optimizer metric, not fitted posterior covariance.
    # Stiffness-informed rounded scales from the mid-course Astra advice.
    scales = [.002, .005, .005, .005, .010, .005, .030, .010, .010,
              .0005, .003, .001, .001, .003, .002, .003, .003, .003,
              .010, .010, .020, .003, .003, .020]
    main(start_checkpoint=BASE / 'r2_conditional_full_gradient78_20261010/gradient.npz',
         max_evaluations=18, max_iterations=17, max_line_search=10,
         maxcor=10, app_seconds=18000, nuisance_scale=scales)
