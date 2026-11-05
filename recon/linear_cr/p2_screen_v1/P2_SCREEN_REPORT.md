# P2a paired Local-Group screen

- Frozen config SHA-256:
  `1c066ae189dee4c077b8400c4427d108f6e7dfc3d6ebedbd1b571cf22d7b3273`
- Parents: 1002 and 1009
- Identical small-scale seeds per parent: 2001--2008
- Forward resolution: \(576^3\) in \(384\,h^{-1}\mathrm{Mpc}\)
- Particle mass: \(2.549\times10^{10}\,M_\odot/h\)
- Compute: Slurm job 187036, A40, completed in 13:34

## Result

No one of the 16 paired combinations passed the frozen P2a pair screen.
This is not a separation, kinematic, or isolation failure: no catalog contains
even one eligible \(5\times10^{11}\)--\(4\times10^{12}\,M_\odot/h\) halo within
6 \(h^{-1}\)Mpc of the observer.

The nearest halos occur at about 10 \(h^{-1}\)Mpc for parent 1002 and
7--8 \(h^{-1}\)Mpc for parent 1009. An independent parent-resolution
measurement confirms that all 16 original posterior members have negative
4 \(h^{-1}\)Mpc-smoothed density at the observer. The two P1 survivors have
especially deep central deficits.

## Interpretation

The current velocity-only parent posterior does not reproduce the Local Sheet
at the observer. Increasing only the number of high-k refinement seeds is not a
valid remedy because all of those seeds share the same underdense coarse
environment.

The v1 catalog likelihood excludes every row with \(cz<1000\) km/s (345 local
rows in the prepared CF4 catalog, including the few grouped representations
of Local-Group galaxies). Even if included, those sparse grouped radial
velocities cannot determine MW--M31--M33 halo-scale phases. The corrected model
must distinguish:

1. CF4 constraints on the large-scale environment;
2. an explicitly documented low-redshift/local-environment likelihood;
3. nonlinear rejection selection on MW--M31--M33 halo properties.

No P2b RAMSES zoom is authorized by this screen result.
