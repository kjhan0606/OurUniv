"""Chain-facing, training-domain field MCSE bookkeeping for R2.

The caller must pass the post-Metropolis retained field, including the exact
previous field after a rejection. This adapter records transition accounting,
checks the frozen training-key/reporting-region identity, and keeps the
predeclared batch scales resumable without storing field draws.
"""
from __future__ import annotations

import hashlib

import numpy as np

from cf4_r2_field_batch_mcse import MultiScaleFieldMCSEAccumulator
from cf4_r2_field_reporting_regions import training_count_cell_region


def validate_field_mcse_schedule(batch_sizes, minimum_batches, retained_capacity):
    """Validate, but never choose, the two predeclared retained-state scales."""
    try:
        sizes = tuple(batch_sizes)
    except TypeError as error:
        raise ValueError("exactly two integer field-MCSE batch sizes are required") from error
    if (len(sizes) != 2 or any(not isinstance(size, (int, np.integer))
            or isinstance(size, (bool, np.bool_)) or int(size) < 2 for size in sizes)
            or len({int(size) for size in sizes}) != 2):
        raise ValueError("exactly two distinct field-MCSE batch sizes >=2 are required")
    if (not isinstance(minimum_batches, (int, np.integer))
            or isinstance(minimum_batches, (bool, np.bool_))
            or int(minimum_batches) < 2):
        raise ValueError("field-MCSE minimum batch count must be an integer >=2")
    if (not isinstance(retained_capacity, (int, np.integer))
            or isinstance(retained_capacity, (bool, np.bool_))
            or int(retained_capacity) < 1):
        raise ValueError("positive retained-state capacity required")
    sizes = tuple(sorted(int(size) for size in sizes))
    minimum_batches = int(minimum_batches)
    retained_capacity = int(retained_capacity)
    for size in sizes:
        if retained_capacity % size:
            raise ValueError("retained chain capacity must end on a complete batch at both scales")
        if retained_capacity // size < minimum_batches:
            raise ValueError("retained chain capacity cannot meet the requested minimum batches")
    return dict(batch_sizes=sizes, minimum_batches=minimum_batches,
        retained_capacity=retained_capacity)


class RetainedFieldMCSEDiagnostics:
    """Online field-map MCSE for one chain, never a convergence verdict."""

    VERSION = 1
    REGION_NAME = "training_count_cells"

    def __init__(self, shape, *, train_keys, batch_sizes, minimum_batches,
                 count_grid_size=128, population_count=6,
                 density_reference=1.0):
        raw_shape = tuple(shape)
        if (len(raw_shape) != 3 or any(not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) for value in raw_shape)):
            raise ValueError("native cubic field shape must use integer dimensions")
        shape = tuple(int(value) for value in raw_shape)
        if len(set(shape)) != 1 or min(shape) < 1:
            raise ValueError("native cubic field shape required")
        if (not isinstance(count_grid_size, (int, np.integer))
                or isinstance(count_grid_size, (bool, np.bool_))
                or not isinstance(population_count, (int, np.integer))
                or isinstance(population_count, (bool, np.bool_))):
            raise ValueError("count-grid and population dimensions must be integers")
        keys = np.asarray(train_keys)
        if keys.ndim != 1 or (keys.size and not np.issubdtype(keys.dtype, np.integer)):
            raise ValueError("frozen training keys must be a one-dimensional integer array")
        keys = np.asarray(keys, dtype=np.int64)
        self.shape = shape
        self.count_grid_size = int(count_grid_size)
        self.population_count = int(population_count)
        self.train_key_sha256 = hashlib.sha256(
            np.ascontiguousarray(keys.astype("<i8", copy=False)).tobytes()).hexdigest()
        region = training_count_cell_region(keys,
            count_grid_size=self.count_grid_size,
            field_grid_size=shape[0], population_count=self.population_count)
        self.region_sha256 = hashlib.sha256(
            np.packbits(region.reshape(-1), bitorder="little").tobytes()).hexdigest()
        self.regions = {self.REGION_NAME: region}
        self.accumulator = MultiScaleFieldMCSEAccumulator(shape,
            batch_sizes=batch_sizes, minimum_batches=minimum_batches,
            density_reference=density_reference)
        self.accepted_transitions = 0
        self.rejected_transitions = 0

    @property
    def retained_states(self):
        return self.accumulator.states

    @property
    def estimated_accumulator_bytes(self):
        return self.accumulator.estimated_accumulator_bytes

    @property
    def batch_sizes(self):
        return self.accumulator.batch_sizes

    @property
    def minimum_batches(self):
        return self.accumulator.minimum_batches

    def update(self, retained_field, *, transition, accepted):
        """Add exactly one post-warmup retained transition state.

        ``retained_field`` is deliberately named to distinguish it from a
        proposal endpoint. For rejected transitions the caller must pass the
        unchanged field associated with the previous retained state.
        """
        if (not isinstance(transition, (int, np.integer))
                or isinstance(transition, (bool, np.bool_))
                or int(transition) != self.retained_states + 1):
            raise ValueError("retained transition indices must be consecutive from one")
        if not isinstance(accepted, (bool, np.bool_)):
            raise ValueError("accepted must be boolean")
        rho = np.asarray(retained_field["rho"], dtype=np.float64)
        velocity = np.asarray(retained_field["mean_velocity_km_s"], dtype=np.float64)
        valid = np.asarray(retained_field["velocity_valid"], dtype=bool)
        if (rho.shape != self.shape or valid.shape != self.shape
                or velocity.shape != (3,) + self.shape):
            raise ValueError("retained density, velocity, and occupancy maps must align")
        if (not np.isfinite(rho).all() or not np.isfinite(velocity).all()
                or np.any(rho < 0) or not np.array_equal(valid, rho > 0)):
            raise ValueError("retained field must be finite, nonnegative, and occupancy-aligned")
        with np.errstate(over="ignore", invalid="ignore"):
            rho2 = rho * rho
            velocity2 = velocity * velocity
        if not np.isfinite(rho2).all() or not np.isfinite(velocity2).all():
            raise ValueError("retained field moments overflow float64")

        self.accumulator.update(dict(rho=rho,
            mean_velocity_km_s=velocity, velocity_valid=valid))
        if accepted:
            self.accepted_transitions += 1
        else:
            self.rejected_transitions += 1

    def checkpoint_arrays(self):
        result = dict(format_version=np.array(self.VERSION, dtype=np.int64),
            shape=np.asarray(self.shape, dtype=np.int64),
            count_grid_size=np.array(self.count_grid_size, dtype=np.int64),
            population_count=np.array(self.population_count, dtype=np.int64),
            train_key_sha256=np.array(self.train_key_sha256),
            region_sha256=np.array(self.region_sha256),
            accepted_transitions=np.array(self.accepted_transitions, dtype=np.int64),
            rejected_transitions=np.array(self.rejected_transitions, dtype=np.int64))
        result.update({f"mcse__{key}": value for key, value
            in self.accumulator.checkpoint_arrays().items()})
        return result

    @classmethod
    def from_checkpoint(cls, state, *, train_keys):
        def scalar_int(name, minimum=0):
            value = np.asarray(state[name])
            if (value.shape or not np.issubdtype(value.dtype, np.integer)
                    or int(value) < minimum):
                raise ValueError(f"invalid {name} checkpoint value")
            return int(value)

        def scalar_text(name):
            value = np.asarray(state[name])
            if value.shape or value.dtype.kind != "U":
                raise ValueError(f"invalid {name} checkpoint value")
            return str(value)

        if scalar_int("format_version", minimum=1) != cls.VERSION:
            raise ValueError("unsupported retained-field MCSE checkpoint version")
        shape = np.asarray(state["shape"])
        if (shape.ndim != 1 or shape.size != 3
                or not np.issubdtype(shape.dtype, np.integer)):
            raise ValueError("invalid retained-field MCSE checkpoint shape")
        count_grid_size = scalar_int("count_grid_size", minimum=1)
        population_count = scalar_int("population_count", minimum=1)
        keys = np.asarray(train_keys)
        if keys.ndim != 1 or (keys.size and not np.issubdtype(keys.dtype, np.integer)):
            raise ValueError("frozen training keys must be a one-dimensional integer array")
        keys = np.asarray(keys, dtype=np.int64)
        key_hash = hashlib.sha256(
            np.ascontiguousarray(keys.astype("<i8", copy=False)).tobytes()).hexdigest()
        if key_hash != scalar_text("train_key_sha256"):
            raise ValueError("training-key identity differs from the field-MCSE checkpoint")

        nested = {key[len("mcse__"):]: value for key, value in state.items()
            if key.startswith("mcse__")}
        accumulator = MultiScaleFieldMCSEAccumulator.from_checkpoint(nested)
        if accumulator.shape != tuple(int(value) for value in shape):
            raise ValueError("field geometry differs from the field-MCSE checkpoint")
        field_shape = tuple(int(value) for value in shape)
        region = training_count_cell_region(keys,
            count_grid_size=count_grid_size, field_grid_size=field_shape[0],
            population_count=population_count)
        region_hash = hashlib.sha256(
            np.packbits(region.reshape(-1), bitorder="little").tobytes()).hexdigest()
        if region_hash != scalar_text("region_sha256"):
            raise ValueError("training reporting region differs from field-MCSE checkpoint")
        result = cls.__new__(cls)
        result.shape = field_shape
        result.count_grid_size = count_grid_size
        result.population_count = population_count
        result.train_key_sha256 = key_hash
        result.region_sha256 = region_hash
        result.regions = {cls.REGION_NAME: region}
        result.accumulator = accumulator
        result.accepted_transitions = scalar_int("accepted_transitions")
        result.rejected_transitions = scalar_int("rejected_transitions")
        if result.accepted_transitions + result.rejected_transitions != result.retained_states:
            raise ValueError("field-MCSE transition accounting is inconsistent")
        return result

    def merge_independent_chains_inplace(self, other_chains):
        """Destructively pool compatible complete-chain sufficient statistics.

        The receiver's training domain and contract are retained. Chain-level
        disagreement must be assessed before this merge; the pooled MCSE is
        not itself evidence of chain convergence.
        """
        if isinstance(other_chains, RetainedFieldMCSEDiagnostics):
            other_chains = (other_chains,)
        else:
            try:
                other_chains = tuple(other_chains)
            except TypeError as error:
                raise ValueError("other chain field-MCSE accumulators required") from error
        if not other_chains or any(not isinstance(chain,
                RetainedFieldMCSEDiagnostics) for chain in other_chains):
            raise ValueError("at least one other retained-field chain is required")
        chains = (self,) + other_chains
        if len({id(chain) for chain in chains}) != len(chains):
            raise ValueError("a retained-field chain cannot be merged more than once")
        contract = (self.shape, self.count_grid_size, self.population_count,
            self.train_key_sha256, self.region_sha256, self.batch_sizes,
            self.minimum_batches)
        for chain in chains:
            candidate = (chain.shape, chain.count_grid_size,
                chain.population_count, chain.train_key_sha256,
                chain.region_sha256, chain.batch_sizes, chain.minimum_batches)
            if candidate != contract:
                raise ValueError("retained-field chain MCSE contracts differ")
            if chain.accepted_transitions + chain.rejected_transitions != chain.retained_states:
                raise ValueError("field-MCSE transition accounting is inconsistent")
        accepted = sum(chain.accepted_transitions for chain in chains)
        rejected = sum(chain.rejected_transitions for chain in chains)
        self.accumulator.merge_independent_chains_inplace(
            [chain.accumulator for chain in other_chains])
        self.accepted_transitions = accepted
        self.rejected_transitions = rejected
        return self

    def finalize_scale(self, batch_size):
        result = self.accumulator.finalize_scale(batch_size, regions=self.regions)
        if result["status"] == "INSUFFICIENT_COMPLETE_BATCHES":
            result["reporting_region_cell_denominators"] = dict(
                full_box=int(np.prod(self.shape)),
                training_count_cells=int(self.regions[self.REGION_NAME].sum()))
        return result

    def summary(self):
        return dict(retained_states=self.retained_states,
            accepted_transitions=self.accepted_transitions,
            rejected_transitions=self.rejected_transitions,
            batch_sizes=self.batch_sizes, minimum_batches=self.minimum_batches,
            pooled_chain_count=self.accumulator.chain_count,
            estimated_accumulator_bytes=self.estimated_accumulator_bytes,
            reporting_region=self.REGION_NAME,
            full_box_cells=int(np.prod(self.shape)),
            reporting_region_cells=int(self.regions[self.REGION_NAME].sum()),
            training_key_sha256=self.train_key_sha256,
            reporting_region_sha256=self.region_sha256,
            interpretation="field MCSE availability only; not chain convergence or posterior validation")
