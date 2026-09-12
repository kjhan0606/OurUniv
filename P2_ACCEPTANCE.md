# P2 preregistered Local-Group acceptance test

P1 leaves two parent realizations, seeds 1002 and 1009. P2 applies the same
eight small-scale phase seeds (2001--2008) to both parents. This paired design
prevents one parent from receiving more chances than the other.

## P2a: full-box screen

Each field is embedded from \(192^3\) to \(576^3\) in the same
\(384\,h^{-1}\mathrm{Mpc}\) periodic box and evolved with the parent cosmology.
The particle mass is approximately \(2.55\times10^{10}\,M_\odot/h\). FoF is used
only to locate plausible pairs in the central region.

A screening pair must satisfy:

- both FoF masses between \(5\times10^{11}\) and
  \(4\times10^{12}\,M_\odot/h\), with mass ratio at most 4;
- separation \(0.3\)--\(1.2\,h^{-1}\mathrm{Mpc}\);
- midpoint within \(5\,h^{-1}\mathrm{Mpc}\) of the CF4 observer;
- no halo above \(5\times10^{12}\,M_\odot/h\) within
  \(3\,h^{-1}\mathrm{Mpc}\).

The PM force scale is comparable to or larger than the observed MW--M31
separation. Radial and tangential velocities therefore affect only the frozen
ranking score, not the P2a hard pass. A possible M33 companion is recorded but
is not required because FoF can merge M31 and M33.

## P2b: definitive zoom gate

Shortlisted candidates are re-simulated with RAMSES and analyzed with the
user's compiled HOP, followed by direct \(M_{200c}\) profiles. The hard gate is:

- both \(M_{200c}=0.5\)--\(3.0\times10^{12}\,M_\odot/h\), mass ratio at most 3;
- separation \(0.45\)--\(0.75\,h^{-1}\mathrm{Mpc}\);
- total radial velocity -200 to -20 km/s and tangential velocity at most
  120 km/s;
- the same midpoint and isolation requirements as P2a, plus no third halo
  more massive than the smaller pair member within \(2.5\,h^{-1}\mathrm{Mpc}\);
- contaminant mass fraction below \(10^{-4}\) inside each \(R_{200c}\);
- the P1 Virgo, Coma, Local Void, and Boötes geometry remains valid after
  placing the observer at the accepted pair midpoint.

M33 requires an unmerged HOP density peak or a RAMSES PHEW cross-check:
\(3\times10^{10}\)--\(5\times10^{11}\,M_\odot/h\), within
0.08--0.35 \(h^{-1}\mathrm{Mpc}\) of M31 and below 30 percent of its mass.
The standard regrouped HOP host catalog by itself is not sufficient evidence
for this subhalo-scale requirement.

Exact machine-readable thresholds and the frozen ranking weights are in
`config/p2_lg_targets_v1.json`.
