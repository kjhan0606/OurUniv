"""Fixed-batch Monte Carlo errors for one chain of Eulerian field states.

The sampler calls ``update`` once per post-Metropolis retained state,
including an unchanged state after rejection. This estimates MC error in the
posterior mean and SD maps from batch means; it does not establish stationarity
or combine chains. Velocity moments are conditional on nonempty cells, with
occupancy reported and diagnosed separately. Input ``rho`` is a nonnegative
density map; ``density_reference`` is subtracted from its posterior mean to
report the density-contrast mean (the R2 PM mesh convention uses reference 1).
"""
from __future__ import annotations

import numpy as np


class FieldBatchMCSEAccumulator:
    """Accumulate map moments and fixed-batch MCSE with constant memory."""

    _PENDING_ARRAYS = (
        "pending_rho", "pending_rho2", "pending_occupied", "pending_v",
        "pending_v2")
    _SUMMARY_ARRAYS = (
        "sum_rho", "sum_rho2", "sum_rho_sq", "sum_rho2_sq",
        "sum_rho_rho2", "sum_occupied", "sum_occupied_sq", "sum_v",
        "sum_v2", "sum_v_sq", "sum_v2_sq", "sum_v_v2",
        "sum_occupied_v", "sum_occupied_v2")

    def __init__(self, shape, *, batch_size, minimum_batches=8,
                 density_reference=1.0):
        shape = tuple(int(x) for x in shape)
        if len(shape) != 3 or len(set(shape)) != 1 or min(shape) < 1:
            raise ValueError("native cubic field shape required")
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("batch size must be an integer >=2")
        if not isinstance(minimum_batches, int) or minimum_batches < 2:
            raise ValueError("at least two complete batches are required")
        density_reference = float(density_reference)
        if not np.isfinite(density_reference) or density_reference <= 0:
            raise ValueError("finite positive density reference required")
        self.shape = shape
        self.batch_size = batch_size
        self.minimum_batches = minimum_batches
        self.density_reference = density_reference
        self.states = 0
        self.pending_states = 0
        self.batches = 0

        def scalar():
            return np.zeros(shape, dtype=np.float64)

        def vector():
            return np.zeros((3,) + shape, dtype=np.float64)

        # Per-state sufficient sums for the batch currently being assembled.
        self.pending_rho = scalar()
        self.pending_rho2 = scalar()
        self.pending_occupied = scalar()
        self.pending_v = vector()
        self.pending_v2 = vector()

        # Cross-products of batch means. These are sufficient for the delta-
        # method MCSEs of E[rho], SD(rho), E[v|occupied], SD(v|occupied), and
        # occupancy without retaining every full field draw.
        self.sum_rho = scalar()
        self.sum_rho2 = scalar()
        self.sum_rho_sq = scalar()
        self.sum_rho2_sq = scalar()
        self.sum_rho_rho2 = scalar()
        self.sum_occupied = scalar()
        self.sum_occupied_sq = scalar()
        self.sum_v = vector()
        self.sum_v2 = vector()
        self.sum_v_sq = vector()
        self.sum_v2_sq = vector()
        self.sum_v_v2 = vector()
        self.sum_occupied_v = vector()
        self.sum_occupied_v2 = vector()

    @property
    def estimated_accumulator_bytes(self):
        """Bytes in fixed-size accumulator arrays, excluding finalize temporaries."""
        # 9 pending and 28 batch-summary full-grid float64 arrays, counting
        # each velocity component as one full-grid array.
        return 37 * int(np.prod(self.shape)) * np.dtype(np.float64).itemsize

    def update(self, field):
        """Add one retained ``rho``, mean-velocity, and occupancy state."""
        rho = np.asarray(field["rho"], dtype=np.float64)
        velocity = np.asarray(field["mean_velocity_km_s"], dtype=np.float64)
        valid = np.asarray(field["velocity_valid"], dtype=bool)
        if (rho.shape != self.shape or valid.shape != self.shape
                or velocity.shape != (3,) + self.shape):
            raise ValueError("aligned native density/velocity/occupancy maps required")
        if (not np.isfinite(rho).all() or not np.isfinite(velocity).all()
                or np.any(rho < 0) or not np.array_equal(valid, rho > 0)):
            raise ValueError("field must be finite, nonnegative, and occupancy-aligned")

        occupied = valid.astype(np.float64)
        # Validate derived moments too: finite values can still overflow when
        # squared. Do not leave a partially updated accumulator on bad input.
        with np.errstate(over="ignore", invalid="ignore"):
            rho2 = rho * rho
            velocity2 = velocity * velocity
        if not np.isfinite(rho2).all() or not np.isfinite(velocity2).all():
            raise ValueError("field moments overflow float64")

        # Validate all inputs before mutating the accumulator.
        self.pending_rho += rho
        self.pending_rho2 += rho2
        self.pending_occupied += occupied
        self.pending_v += velocity * occupied[None]
        self.pending_v2 += velocity2 * occupied[None]
        self.pending_states += 1
        self.states += 1
        if self.pending_states == self.batch_size:
            self._finish_batch()

    def _finish_batch(self):
        if self.pending_states != self.batch_size:
            raise RuntimeError("only a complete fixed batch can be committed")
        inv = 1.0 / self.batch_size
        rho = self.pending_rho * inv
        rho2 = self.pending_rho2 * inv
        occupied = self.pending_occupied * inv
        velocity = self.pending_v * inv
        velocity2 = self.pending_v2 * inv

        self.sum_rho += rho
        self.sum_rho2 += rho2
        self.sum_rho_sq += rho * rho
        self.sum_rho2_sq += rho2 * rho2
        self.sum_rho_rho2 += rho * rho2
        self.sum_occupied += occupied
        self.sum_occupied_sq += occupied * occupied
        self.sum_v += velocity
        self.sum_v2 += velocity2
        self.sum_v_sq += velocity * velocity
        self.sum_v2_sq += velocity2 * velocity2
        self.sum_v_v2 += velocity * velocity2
        self.sum_occupied_v += occupied[None] * velocity
        self.sum_occupied_v2 += occupied[None] * velocity2

        self.batches += 1
        self.pending_states = 0
        for name in ("pending_rho", "pending_rho2", "pending_occupied",
                     "pending_v", "pending_v2"):
            getattr(self, name).fill(0.)

    def checkpoint_arrays(self):
        """Expose sufficient statistics for an exact, lossless chain resume."""
        result = dict(format_version=np.array(1, dtype=np.int64),
            batch_size=np.array(self.batch_size, dtype=np.int64),
            minimum_batches=np.array(self.minimum_batches, dtype=np.int64),
            density_reference=np.array(self.density_reference, dtype=np.float64),
            states=np.array(self.states, dtype=np.int64),
            pending_states=np.array(self.pending_states, dtype=np.int64),
            batches=np.array(self.batches, dtype=np.int64))
        result.update({name: getattr(self, name) for name in
            self._PENDING_ARRAYS + self._SUMMARY_ARRAYS})
        return result

    @classmethod
    def from_checkpoint(cls, state):
        """Restore a checkpoint without duplicating its full-grid arrays."""
        def scalar_int(name, minimum=0):
            value = np.asarray(state[name])
            if (value.shape or not np.issubdtype(value.dtype, np.integer)
                    or int(value) < minimum):
                raise ValueError(f"invalid {name} checkpoint scalar")
            return int(value)

        if scalar_int("format_version", minimum=1) != 1:
            raise ValueError("unsupported field-MCSE checkpoint version")
        batch_size = scalar_int("batch_size", minimum=2)
        minimum_batches = scalar_int("minimum_batches", minimum=2)
        density_reference = np.asarray(state["density_reference"])
        if (density_reference.shape or density_reference.dtype != np.dtype(np.float64)
                or not np.isfinite(density_reference)):
            raise ValueError("invalid density reference checkpoint scalar")
        states = scalar_int("states")
        pending_states = scalar_int("pending_states")
        batches = scalar_int("batches")
        if pending_states >= batch_size or states != batches * batch_size + pending_states:
            raise ValueError("inconsistent field-MCSE checkpoint counters")

        first = np.asarray(state["pending_rho"])
        if first.ndim != 3 or len(set(first.shape)) != 1 or min(first.shape) < 1:
            raise ValueError("invalid field-MCSE checkpoint shape")
        result = cls(first.shape, batch_size=batch_size,
            minimum_batches=minimum_batches,
            density_reference=float(density_reference))
        for name in result._PENDING_ARRAYS + result._SUMMARY_ARRAYS:
            value = np.asarray(state[name])
            target = getattr(result, name)
            if (value.shape != target.shape or value.dtype != np.dtype(np.float64)
                    or not np.isfinite(value).all()):
                raise ValueError(f"invalid field-MCSE checkpoint array {name}")
            setattr(result, name, value)
        if np.any(result.pending_occupied < 0) or np.any(result.pending_occupied > pending_states):
            raise ValueError("invalid pending occupancy checkpoint")
        if np.any(result.sum_occupied < 0) or np.any(result.sum_occupied > batches):
            raise ValueError("invalid batch occupancy checkpoint")
        result.states = states
        result.pending_states = pending_states
        result.batches = batches
        return result

    @staticmethod
    def _sample_covariance(sum_a, sum_b, sum_ab, n):
        return (sum_ab - sum_a * sum_b / n) / (n - 1)

    @staticmethod
    def _nonnegative_variance(value, scale, label):
        tolerance = 1e-12 * np.maximum(1., scale)
        if np.any(value < -tolerance):
            raise FloatingPointError(f"negative {label} beyond roundoff")
        return np.maximum(value, 0.)

    @staticmethod
    def _ratio(mcse, posterior_sd):
        result = np.full_like(posterior_sd, np.nan, dtype=np.float64)
        finite_sd = np.isfinite(posterior_sd)
        positive = finite_sd & (posterior_sd > 0)
        result[positive] = mcse[positive] / posterior_sd[positive]
        zero = finite_sd & (posterior_sd == 0)
        result[zero & (mcse == 0)] = 0.
        result[zero & (mcse > 0)] = np.inf
        return result

    def _region_summary(self, ratio, defined, posterior_sd, regions):
        summaries = {}
        for name, region in regions.items():
            mask = np.asarray(region)
            if mask.dtype != np.bool_ or mask.shape != self.shape:
                raise ValueError(f"region {name!r} must be an aligned boolean map")
            selected = ratio[mask]
            selected_defined = defined[mask]
            selected_sd = posterior_sd[mask]
            defined_values = selected[selected_defined]
            finite = defined_values[np.isfinite(defined_values)]
            summaries[name] = dict(
                region_cells=int(mask.sum()),
                defined_cells=int(selected_defined.sum()),
                undefined_cells=int(mask.sum() - selected_defined.sum()),
                finite_ratio_cells=int(finite.size),
                nonfinite_ratio_cells=int(selected_defined.sum() - finite.size),
                zero_posterior_sd_cells=int(np.sum(
                    selected_defined & (selected_sd == 0))),
                nonfinite_posterior_sd_cells=int(np.sum(
                    selected_defined & ~np.isfinite(selected_sd))),
                p50=float(np.percentile(finite, 50)) if finite.size else None,
                p95=float(np.percentile(finite, 95)) if finite.size else None,
                p99=float(np.percentile(finite, 99)) if finite.size else None,
                max_finite=float(np.max(finite)) if finite.size else None)
        return summaries

    def finalize(self, *, regions=None):
        """Return posterior maps and MCSE maps after complete fixed batches.

        Zero-SD cells with nonzero MCSE retain an infinite ratio and are
        counted; undefined conditional-velocity cells remain explicit rather
        than disappearing from regional denominators.
        """
        if self.pending_states:
            raise ValueError("retained states end in an incomplete fixed batch")
        if self.batches < self.minimum_batches:
            raise ValueError("too few complete batches for MCSE estimation")
        if regions is None:
            regions = {"full_box": np.ones(self.shape, dtype=bool)}
        elif "full_box" not in regions:
            regions = {"full_box": np.ones(self.shape, dtype=bool), **regions}

        b = self.batches
        n = self.states
        rho_mean = self.sum_rho / b
        rho_second = self.sum_rho2 / b
        rho_var_pop = self._nonnegative_variance(
            rho_second - rho_mean * rho_mean,
            np.maximum(rho_second, rho_mean * rho_mean), "density variance")
        rho_sd = np.sqrt(rho_var_pop * n / (n - 1))
        cov_rho = self._sample_covariance(
            self.sum_rho, self.sum_rho, self.sum_rho_sq, b)
        mcse_rho_mean = np.sqrt(self._nonnegative_variance(
            cov_rho, np.abs(cov_rho), "density batch-mean variance") / b)
        cov_rho2 = self._sample_covariance(
            self.sum_rho2, self.sum_rho2, self.sum_rho2_sq, b)
        cov_rho_rho2 = self._sample_covariance(
            self.sum_rho, self.sum_rho2, self.sum_rho_rho2, b)
        # Influence function for variance: rho^2 - 2 E[rho]*rho + constant.
        var_if = self._nonnegative_variance(
            cov_rho2 + 4 * rho_mean * rho_mean * cov_rho
            - 4 * rho_mean * cov_rho_rho2,
            np.abs(cov_rho2) + 4 * rho_mean * rho_mean * np.abs(cov_rho)
            + 4 * np.abs(rho_mean * cov_rho_rho2), "density variance MCSE")
        mcse_rho_var = (n / (n - 1.)) * np.sqrt(var_if / b)
        mcse_rho_sd = np.full_like(rho_sd, np.inf)
        positive_rho_sd = rho_sd > 0
        mcse_rho_sd[positive_rho_sd] = (
            mcse_rho_var[positive_rho_sd] / (2 * rho_sd[positive_rho_sd]))
        mcse_rho_sd[(rho_sd == 0) & (mcse_rho_var == 0)] = 0.

        occupancy = self.sum_occupied / b
        cov_occ = self._sample_covariance(
            self.sum_occupied, self.sum_occupied, self.sum_occupied_sq, b)
        mcse_occupancy = np.sqrt(self._nonnegative_variance(
            cov_occ, np.abs(cov_occ), "occupancy batch-mean variance") / b)
        occupancy_sd = np.sqrt(
            occupancy * (1. - occupancy) * n / (n - 1))
        occupancy_ratio = self._ratio(mcse_occupancy, occupancy_sd)

        velocity_mean = np.full((3,) + self.shape, np.nan, dtype=np.float64)
        velocity_sd = np.full_like(velocity_mean, np.nan)
        velocity_mean_mcse = np.full_like(velocity_mean, np.nan)
        velocity_sd_mcse = np.full_like(velocity_mean, np.nan)
        velocity_ratio = np.full_like(velocity_mean, np.nan)
        velocity_defined = np.broadcast_to(occupancy > 0, velocity_mean.shape).copy()
        velocity_variance_defined = np.broadcast_to(
            occupancy * n > 1, velocity_mean.shape).copy()

        cov_pp = cov_occ
        for axis in range(3):
            y = self.sum_v[axis] / b
            x = self.sum_v2[axis] / b
            cov_yy = self._sample_covariance(
                self.sum_v[axis], self.sum_v[axis], self.sum_v_sq[axis], b)
            cov_xx = self._sample_covariance(
                self.sum_v2[axis], self.sum_v2[axis], self.sum_v2_sq[axis], b)
            cov_yx = self._sample_covariance(
                self.sum_v[axis], self.sum_v2[axis], self.sum_v_v2[axis], b)
            cov_py = self._sample_covariance(
                self.sum_occupied, self.sum_v[axis], self.sum_occupied_v[axis], b)
            cov_px = self._sample_covariance(
                self.sum_occupied, self.sum_v2[axis], self.sum_occupied_v2[axis], b)
            active = occupancy > 0
            mu = np.zeros(self.shape, dtype=np.float64)
            mu[active] = y[active] / occupancy[active]
            variance = np.full(self.shape, np.nan, dtype=np.float64)
            variance[active] = x[active] / occupancy[active] - mu[active] ** 2
            variance[active] = self._nonnegative_variance(
                variance[active], np.maximum(
                    np.abs(x[active] / occupancy[active]), mu[active] ** 2),
                "conditional velocity variance")
            count = occupancy * n
            sample_var = np.full(self.shape, np.nan, dtype=np.float64)
            enough = count > 1
            sample_var[enough] = variance[enough] * count[enough] / (count[enough] - 1)
            sample_sd = np.sqrt(sample_var)

            mean_if_var = np.full(self.shape, np.nan, dtype=np.float64)
            mean_if_var[active] = (
                cov_yy[active] - 2 * mu[active] * cov_py[active]
                + mu[active] ** 2 * cov_pp[active]) / occupancy[active] ** 2
            mean_if_var[active] = self._nonnegative_variance(
                mean_if_var[active], np.abs(cov_yy[active])
                + 2 * np.abs(mu[active] * cov_py[active])
                + mu[active] ** 2 * np.abs(cov_pp[active]),
                "velocity-mean MCSE")

            var_if_var = np.full(self.shape, np.nan, dtype=np.float64)
            c_p = mu * mu - np.nan_to_num(variance, nan=0.)
            # The reported conditional SD uses the unbiased sample variance
            # over the M=n*p occupied retained states. Include the derivative
            # of M/(M-1) with respect to p in its batch-delta MCSE.
            count = occupancy * n
            correction = np.full(self.shape, np.nan, dtype=np.float64)
            correction[enough] = count[enough] / (count[enough] - 1.)
            d_p = np.full(self.shape, np.nan, dtype=np.float64)
            d_y = np.full(self.shape, np.nan, dtype=np.float64)
            d_x = np.full(self.shape, np.nan, dtype=np.float64)
            d_p[enough] = (
                correction[enough] * c_p[enough] / occupancy[enough]
                - n * variance[enough] / (count[enough] - 1.) ** 2)
            d_y[enough] = (-2. * correction[enough] * mu[enough]
                           / occupancy[enough])
            d_x[enough] = correction[enough] / occupancy[enough]
            var_if_var[enough] = (
                d_p[enough] ** 2 * cov_pp[enough]
                + d_y[enough] ** 2 * cov_yy[enough]
                + d_x[enough] ** 2 * cov_xx[enough]
                + 2. * d_p[enough] * d_y[enough] * cov_py[enough]
                + 2. * d_p[enough] * d_x[enough] * cov_px[enough]
                + 2. * d_y[enough] * d_x[enough] * cov_yx[enough])
            var_if_var[enough] = self._nonnegative_variance(
                var_if_var[enough], (
                    d_p[enough] ** 2 * np.abs(cov_pp[enough])
                    + d_y[enough] ** 2 * np.abs(cov_yy[enough])
                    + d_x[enough] ** 2 * np.abs(cov_xx[enough])
                    + 2. * np.abs(d_p[enough] * d_y[enough] * cov_py[enough])
                    + 2. * np.abs(d_p[enough] * d_x[enough] * cov_px[enough])
                    + 2. * np.abs(d_y[enough] * d_x[enough] * cov_yx[enough])),
                "velocity-variance MCSE")

            mean_mcse = np.full(self.shape, np.nan, dtype=np.float64)
            mean_mcse[active] = np.sqrt(mean_if_var[active] / b)
            variance_mcse = np.full(self.shape, np.nan, dtype=np.float64)
            variance_mcse[enough] = np.sqrt(var_if_var[enough] / b)
            sd_mcse = np.full(self.shape, np.nan, dtype=np.float64)
            sd_positive = enough & (sample_sd > 0)
            sd_mcse[sd_positive] = variance_mcse[sd_positive] / (2 * sample_sd[sd_positive])
            sd_mcse[enough & (sample_sd == 0) & (variance_mcse == 0)] = 0.

            velocity_mean[axis, active] = mu[active]
            velocity_sd[axis] = sample_sd
            velocity_mean_mcse[axis] = mean_mcse
            velocity_sd_mcse[axis] = sd_mcse
            velocity_ratio[axis] = self._ratio(mean_mcse, sample_sd)

        density_ratio = self._ratio(mcse_rho_mean, rho_sd)
        density_sd_ratio = self._ratio(mcse_rho_sd, rho_sd)
        density_defined = np.ones(self.shape, dtype=bool)
        velocity_sd_ratio = self._ratio(velocity_sd_mcse, velocity_sd)
        report = dict(retained_states=n, complete_batches=b,
            batch_size=self.batch_size, pending_states=0,
            estimated_accumulator_bytes=self.estimated_accumulator_bytes,
            density_mean=rho_mean, density_posterior_sd=rho_sd,
            density_contrast_mean=rho_mean - self.density_reference,
            density_contrast_posterior_sd=rho_sd,
            density_reference=self.density_reference,
            density_mean_mcse=mcse_rho_mean, density_sd_mcse=mcse_rho_sd,
            density_contrast_mean_mcse=mcse_rho_mean,
            density_contrast_sd_mcse=mcse_rho_sd,
            density_mean_mcse_over_sd=density_ratio,
            density_sd_mcse_over_sd=density_sd_ratio,
            velocity_conditional_mean=velocity_mean,
            velocity_conditional_posterior_sd=velocity_sd,
            velocity_mean_mcse=velocity_mean_mcse,
            velocity_sd_mcse=velocity_sd_mcse,
            velocity_mean_mcse_over_sd=velocity_ratio,
            velocity_sd_mcse_over_sd=velocity_sd_ratio,
            velocity_occupancy_fraction=occupancy,
            velocity_occupancy_mcse=mcse_occupancy,
            velocity_occupancy_posterior_sd=occupancy_sd,
            velocity_occupancy_mcse_over_sd=occupancy_ratio,
            velocity_defined=velocity_defined,
            velocity_variance_defined=velocity_variance_defined)
        report["regional_summary"] = dict(
            density_mean_mcse_over_sd=self._region_summary(
                density_ratio, density_defined, rho_sd, regions),
            density_sd_mcse_over_sd=self._region_summary(
                density_sd_ratio, density_defined, rho_sd, regions),
            velocity_mean_mcse_over_sd=[
                self._region_summary(velocity_ratio[i], velocity_variance_defined[i],
                    velocity_sd[i], regions)
                for i in range(3)],
            velocity_sd_mcse_over_sd=[
                self._region_summary(velocity_sd_ratio[i], velocity_variance_defined[i],
                    velocity_sd[i], regions)
                for i in range(3)],
            velocity_occupancy_mcse_over_sd=self._region_summary(
                occupancy_ratio, np.ones(self.shape, dtype=bool), occupancy_sd, regions))
        return report
