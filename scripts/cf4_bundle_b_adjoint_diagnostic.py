"""Trace-only PMWD failure diagnosis; no simulation or installed-library edit."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from cf4_z0_pm_bridge import make_forward

program=json.loads(Path("config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())
forward,_=make_forward(program)
print("Tracing PM adjoint; no numerical N-body execution",flush=True)
jax.make_jaxpr(jax.grad(lambda x: jnp.sum(forward(x)[0]**2)))(jax.ShapeDtypeStruct((64**3,),jnp.float64))
print("TRACE_PASS",flush=True)
