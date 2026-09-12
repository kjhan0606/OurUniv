#!/usr/bin/env python3
"""Write frozen RAMSES preflight and z=0 namelists for one promoted LG IC."""
from __future__ import annotations

import argparse
from pathlib import Path


RUN = """&RUN_PARAMS
cosmo=.true.
pic=.true.
poisson=.true.
hydro=.false.
sink=.false.
sidm=.false.
nrestart=0
nstepmax={nstepmax}
nremap={nremap}
nsubcycle=1,1,9*2
ncontrol=1
ordering='ksection'
memory_balance=.true.
use_fftw=.true.
ksec_level_balance_alpha=0.5
ksec_level_min_fraction=0.03
ksec_level_bins=256
jobcontrolfile='jobcontrol.txt'
verbose=.false.
/
"""


def namelist(seed: int, ic: Path, *, preflight: bool) -> str:
    outputs = ("noutput=1\naout=1.0\nfoutput=1000000\nfbackup=1000000"
               if preflight else
               "noutput=7\naout=0.05,0.10,0.20,0.3333333333,0.50,0.6666666667,1.0\n"
               "foutput=1000000\nfbackup=1000")
    run = RUN.format(nstepmax=2 if preflight else 10000000,
                     nremap=1 if preflight else 5)
    init_lines = "\n".join(
        f"initfile({level - 8})='{ic}/level_{level:03d}'"
        for level in range(9, 13))
    return f"""! Auto-frozen OurUniv p3429/s{seed} L12/L19 {'preflight' if preflight else 'z=0 pilot'}.
{run}
&OUTPUT_PARAMS
{outputs}
/

&INIT_PARAMS
filetype='grafic'
{init_lines}
/

&AMR_PARAMS
levelmin=9
levelmax=19
nexpand=1,1,9*1
ngridtot=120000000
nparttot=220000000
/

&COSMO_PARAMS
omega_b=0.0
omega_m=0.31
omega_l=0.69
h0=0.746
/

&REFINE_PARAMS
m_refine=11*8.
ivar_refine=0
mass_cut_refine=2.9103830457d-11
interpol_var=1
interpol_type=0
q_refine_holdback=.false.
dr_proper=0.0
/

&POISSON_PARAMS
epsilon=1.d-4
/
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ic-link", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if len(str(args.ic_link)) > 55:
        parser.error("IC link is too long for the RAMSES CHARACTER(LEN=80) path")
    args.outdir.mkdir(parents=True, exist_ok=True)
    preflight = args.outdir / f"ramses_lg_p3429_s{args.seed}_preflight.nml"
    z0 = args.outdir / f"ramses_lg_p3429_s{args.seed}_z0.nml"
    preflight.write_text(namelist(args.seed, args.ic_link, preflight=True))
    z0.write_text(namelist(args.seed, args.ic_link, preflight=False))
    print(preflight)
    print(z0)


if __name__ == "__main__":
    main()
