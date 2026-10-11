"""One fixed-state count-conditioned versus independent-ray CF4 mark control."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_linked_fp_sparse_train import (
    FP, SOURCE, load_train_singletons, select_training_single_mark_links,
)
from cf4_r2_native_to_count_cells import native_moments_to_count_cells
from cf4_r2_observed_ray import observed_ray_components
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_raw_live_mark import (
    POPULATION_ORIGIN, POPULATION_SCALE, chunk_log_terms,
)
from cf4_r2_raw_volume_target import (
    FreshRawSupport, raw_field_logpdf, tracer_geometry, tracer_masses, volume_rule,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
STATE = BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz'
CONTROL_PGC = 1085367
BOX, N, SOURCE_SPACING = 384., 128, 3.
BLOCK, EPS = 256, 1.e-4
CORE_SIGMA, BROAD_SCALE, BROAD_FRACTION = 30., .5, .5
_FAILURE_STATE = None


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source_commit(expected_commit):
    def resolve(revision):
        return subprocess.check_output(
            ['git', 'rev-parse', '--verify', f'{revision}^{{commit}}'],
            cwd=ROOT, text=True).strip()

    expected_revision = resolve(expected_commit)
    source_revision = resolve('HEAD')
    if source_revision != expected_revision:
        raise RuntimeError(
            f'source commit mismatch: {source_revision} != {expected_revision}')
    return source_revision


def components(packs, population):
    pack = packs[population]
    active = np.asarray(pack['mask'], dtype=bool)
    return set(zip(np.asarray(pack['ids'])[active].tolist(),
                   np.asarray(pack['node'])[active].tolist(),
                   np.asarray(pack['bin'])[active].tolist()))


def support_components_by_scale(support_by_scale, population):
    return {scale: components(packs, population)
            for scale, packs in support_by_scale.items()}


def radius_summary(weight, radius):
    weight, radius = map(lambda x: np.asarray(x, dtype=np.float64), (weight, radius))
    total = float(weight.sum())
    if not np.isfinite(weight).all() or np.any(weight < 0) or total <= 0:
        raise FloatingPointError('invalid or empty source mixture')
    probability = weight/total
    order = np.argsort(radius)
    cumulative = np.cumsum(probability[order])
    quantiles = {str(q): float(radius[order[min(
        np.searchsorted(cumulative, q), len(order)-1)]]) for q in (.1, .5, .9)}
    return dict(total_unnormalized_weight=total,
        effective_source_components=float(1./np.sum(probability**2)),
        mean_true_radius_cMpc_h=float(np.sum(probability*radius)),
        true_radius_quantiles_cMpc_h=quantiles)


def main():
    global _FAILURE_STATE
    jax.config.update('jax_enable_x64', True)
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('submit this fixed-state diagnostic to Slurm with a GPU')
    expected_commit = os.environ.get('CF4_EXPECTED_COMMIT')
    if not expected_commit:
        raise RuntimeError('CF4_EXPECTED_COMMIT must pin the submitted source')
    source_commit = verify_source_commit(expected_commit)
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', phase='INPUTS', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=source_commit,
        classification='ONE_TRAINING_LINK_COUNT_FP_OWNERSHIP_CONTROL_NOT_POSTERIOR',
        control_PGC=CONTROL_PGC,
        control_selection_rule='fixed_predeclared_PGC', N=N, box_cMpc_h=BOX,
        field_state='saved N128 accepted pilot; fixed and nonstationary',
        count_occurrence_scored_once=True, heldout_values_read=False,
        PM_evolutions=0, field_fit=False, optimizer_steps=0, sampler=False,
        posterior_promoted=False, R2_complete=False,
        selection_association_calibrated=False,
        multi_member_FP_covariance_calibrated=False,
        MW_M31='same-new-field roles remain ambiguous; no native truth identity used',
        M33='unresolved; later observables must constrain the same NEW field at <=0.3 cMpc/h')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss/1024**2
        temp = out/'result.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(out/'result.json')

    _FAILURE_STATE = (report, save)
    save()
    try:
        _, _, _, _, source, mix, observation, base_geometry = load_inputs()
        options, point, _ = load_train_singletons(SPLIT)
        with np.load(FP, allow_pickle=False) as f:
            options = select_training_single_mark_links(
                options, f['membership_state'].astype(str), include_grouped=True)
            chosen = [item for pop in range(6) for item in options
                      if int(point['population'][item[2]]) == pop]
            registered_pgc = np.asarray([f['PGC'][item[3]] for item in chosen], dtype=np.int64)
            fp_direction, membership = f['directions'].copy(), f['membership_state'].astype(str)
        np.testing.assert_array_equal(mix['PGC'], registered_pgc)
        control_pgc = CONTROL_PGC
        locations = np.flatnonzero(registered_pgc == control_pgc)
        if locations.size != 1:
            raise ValueError('predeclared control must occur once in the v6 training links')
        row = int(locations[0]); link = chosen[row]; point_index = int(link[2])
        population = int(point['population'][point_index])
        flat_cell = int(point['flat_cell'][point_index])
        voxel = np.asarray(np.unravel_index(flat_cell, (N,)*3), dtype=np.int32)
        if int(mix['population'][row]) != population:
            raise ValueError('linked count and FP population keys disagree')
        if not np.array_equal(np.asarray(observation['voxel'][row]), voxel):
            raise ValueError('linked count and FP observed voxels disagree')
        count_key = population*N**3+flat_cell
        with np.load(SPLIT, allow_pickle=False) as f:
            if not np.isin(count_key, f['train_keys']):
                raise ValueError('control count key is not training-owned in v6')

        obs = {key: jnp.asarray(np.asarray(value[row]))
               for key, value in observation.items()}
        obs_batch = {key: np.asarray(value[row:row+1])
                     for key, value in observation.items()}
        point_radius = float(point['radius_cMpc_h'][point_index])
        group_radius = float(observation['radius'][row])
        if not (5. <= point_radius <= 180.) or group_radius <= 0.:
            raise ValueError('control radii outside the count/FP radial support')

        with np.load(STATE, allow_pickle=False) as f:
            rho0, mean_velocity, physical_variance = map(jnp.asarray,
                (f['rho'], f['mean_velocity_km_s'], f['physical_velocity_variance_km2_s2']))
            old_tracer = np.asarray(f['tracer'])
            population_white = jnp.asarray(f['population_white'])
        tracer = jnp.asarray(np.r_[old_tracer[:6], 0.,
                                   .12*np.exp(.5*old_tracer[7]), old_tracer[8]])
        density, velocity_grid, variance_grid = native_moments_to_count_cells(
            rho0, mean_velocity, physical_variance, BOX)
        velocity = jnp.moveaxis(velocity_grid, 0, -1).reshape(-1, 3)
        variance = jnp.moveaxis(variance_grid, 0, -1).reshape(-1, 3)
        source = {key: jnp.asarray(value) for key, value in source.items()}
        if (density.shape != (N,)*3 or velocity.shape != source['positions'].shape
                or variance.shape != velocity.shape):
            raise ValueError('saved N128 field and source-cell geometry do not align')
        rates = tracer_masses(density, tracer)
        geometry = tracer_geometry(tracer, base_geometry)
        parameters = jnp.asarray(POPULATION_ORIGIN)+jnp.asarray(POPULATION_SCALE)*population_white
        closure = lambda x: dict(core_sigma_km_s=CORE_SIGMA,
            dispersion_scale=BROAD_SCALE*jnp.exp(x), broad_fraction=BROAD_FRACTION)

        report.update(phase='REGISTERED_CONTROL',
            link=dict(PGC=control_pgc, source_group_label=str(link[0]),
                count_population=population, count_voxel_ijk=voxel.tolist(),
                count_key=count_key, linked_point_index=point_index,
                point_individual_radius_cMpc_h=point_radius,
                CF4_group_radius_cMpc_h=group_radius,
                CF4_minus_2Mpp_radius_cMpc_h=group_radius-point_radius,
                membership_state=membership[int(link[3])],
                role='training only; not truth-selected'),
            kernel=dict(count_source='existing voxel-CDF source-selected RSD kernel; GL2 source volume',
                raw_FP='existing optical/K mark numerator and selection denominator, applied once',
                point_redshift_conditioned_separately=True, extra_count_factor=False,
                extra_fn=False, d2_rho_prior=False, periodic_voxel_images=27,
                LOS_scale_direction='log of broad diagonal-moment scale; +/-1e-4'))
        save()

        offsets, volume_weights = volume_rule(SOURCE_SPACING, 2)
        def builder(source_radius=None):
            kwargs = {} if source_radius is None else dict(
                source_conditioning_radius_cMpc_h=np.asarray([source_radius]))
            return FreshRawSupport(np.asarray(source['positions']),
                np.asarray(source['angular']), np.asarray([population]), obs_batch,
                base_geometry, source_spacing=SOURCE_SPACING, volume_order=2,
                block=BLOCK, **kwargs)

        def build_support(support_builder, logscale):
            return support_builder.build(velocity, tracer,
                velocity_variances=variance, velocity_closure=closure(jnp.asarray(logscale)))

        # Prove the current default path equals an explicit CF4-radius condition.
        current_builder = builder()
        current_packs, current_support = build_support(current_builder, 0.)
        raw_kwargs = dict(source_spacing=SOURCE_SPACING, volume_order=2,
            block=BLOCK, cut_order=64, velocity_variances=variance,
            velocity_closure=closure(jnp.asarray(0.)))
        current_default = raw_field_logpdf(density, velocity, tracer, population_white,
            current_packs, source, {k: jnp.asarray(v) for k, v in obs_batch.items()},
            base_geometry, **raw_kwargs)
        current_explicit = raw_field_logpdf(density, velocity, tracer, population_white,
            current_packs, source, {k: jnp.asarray(v) for k, v in obs_batch.items()},
            base_geometry, source_conditioning_radius_cMpc_h=jnp.asarray([group_radius]),
            **raw_kwargs)
        current_value = float(current_default[0])
        default_identity_error = float(abs(current_value-float(current_explicit[0])))
        if not np.isfinite([current_value, default_identity_error]).all() or default_identity_error > 2e-6:
            raise AssertionError('explicit current-radius kernel differs from existing raw target')

        # A larger-radius support is a finite local superset; verify it contains
        # nominal and minus-step candidates before taking the derivative.
        point_builder = builder(point_radius)
        support_by_scale = {}
        info_by_scale = {}
        for value in (-EPS, 0., EPS):
            support_by_scale[value], info_by_scale[value] = build_support(point_builder, value)
        support_sets = support_components_by_scale(support_by_scale, population)
        if not support_sets[-EPS] <= support_sets[EPS] or not support_sets[0.] <= support_sets[EPS]:
            raise ValueError('broad +epsilon support is not a superset of nearby count kernels')
        point_packs = support_by_scale[EPS]

        def count_mark(logscale, packs=point_packs):
            return raw_field_logpdf(density, velocity, tracer, population_white,
                packs, source, {k: jnp.asarray(v) for k, v in obs_batch.items()},
                base_geometry, source_conditioning_radius_cMpc_h=jnp.asarray([point_radius]),
                **dict(raw_kwargs, velocity_closure=closure(logscale)))[0]

        count_value, count_ad = jax.jit(jax.value_and_grad(
            lambda x: count_mark(x)))(jnp.asarray(0.))
        count_fd = float((count_mark(jnp.asarray(EPS))-count_mark(jnp.asarray(-EPS)))/(2*EPS))
        count_value, count_ad = float(count_value), float(count_ad)
        count_error = abs(count_ad-count_fd)/max(1., abs(count_ad), abs(count_fd))
        if not np.isfinite([count_value, count_ad, count_fd]).all() or count_error > 2e-5:
            raise AssertionError('count-conditioned raw-FP LOS-scale AD/FD failed')

        # Summarize the pre-mark source mixture from the same count kernel.
        pack = point_packs[population]
        active = np.asarray(pack['mask'], dtype=bool)
        ids = np.asarray(pack['ids'])[active]
        nodes = np.asarray(pack['node'])[active]
        bins = np.asarray(pack['bin'])[active]
        ids_j, nodes_j = jnp.asarray(ids), jnp.asarray(nodes)
        positions = (source['positions'][ids_j]+jnp.asarray(offsets)[nodes_j]) % BOX
        kernel_mass = point_builder.mixed_weight(velocity[ids_j], positions,
            source['angular'][:, ids_j], jnp.asarray(voxel), jnp.asarray(point_radius),
            population, variance[ids_j], jnp.asarray(CORE_SIGMA),
            jnp.asarray(BROAD_SCALE), jnp.asarray(BROAD_FRACTION), tracer)
        subcell_rate = rates[:, ids_j]*jnp.asarray(volume_weights)[nodes_j]
        count_mass = np.asarray(kernel_mass*subcell_rate)
        count_weight = count_mass[bins, np.arange(len(bins))]
        rel = (np.asarray(source['positions'])[ids]+offsets[nodes]
               -BOX/2.+BOX/2.) % BOX-BOX/2.
        count_true_radius = np.linalg.norm(rel, axis=1)
        count_mixture = radius_summary(count_weight, count_true_radius)

        direction = jnp.asarray(fp_direction[int(link[3])])
        def ray_terms(logscale):
            pos, ray_velocity, _, angular, q, radial_mass = observed_ray_components(
                direction, obs['radius'], velocity, variance, rates, source['angular'],
                population, geometry, closure(logscale), source_grid=N, order=8, segments=4)
            return chunk_log_terms(parameters, pos, ray_velocity,
                jnp.ones((5, q.size)), angular, obs, population=population,
                geometry=geometry, cut_order=64, radial_source_mass=radial_mass,
                source_radius_cMpc_h=q)

        def ray_mark(logscale):
            numerator, denominator = ray_terms(logscale)
            return numerator-denominator
        ray0 = observed_ray_components(direction, obs['radius'], velocity, variance,
            rates, source['angular'], population, geometry, closure(jnp.asarray(0.)),
            source_grid=N, order=8, segments=4)
        ray_q, ray_mass = np.asarray(ray0[4]), np.asarray(ray0[5])
        ray_value, ray_ad = jax.jit(jax.value_and_grad(ray_mark))(jnp.asarray(0.))
        ray_fd = float((ray_mark(jnp.asarray(EPS))-ray_mark(jnp.asarray(-EPS)))/(2*EPS))
        ray_value, ray_ad = float(ray_value), float(ray_ad)
        ray_error = abs(ray_ad-ray_fd)/max(1., abs(ray_ad), abs(ray_fd))
        if not np.isfinite([ray_value, ray_ad, ray_fd]).all() or ray_error > 2e-5:
            raise AssertionError('independent-ray raw-FP LOS-scale AD/FD failed')
        face = BOX/2./float(np.max(np.abs(np.asarray(direction))))
        ray_image_fraction = float(ray_mass[:, ray_q > face].sum()/ray_mass.sum())
        ray_mixture = radius_summary(ray_mass.sum(axis=0), ray_q)
        image_margin = BOX/2.-point_radius
        if image_margin <= 0.:
            raise ValueError('count-conditioned radial shell has no central-image margin')

        report.update(status='COUNT_FP_OWNERSHIP_CONTROL_PASS_NOT_POSTERIOR',
            current_target_identity=dict(existing_raw_logfactor=current_value,
                explicit_same_radius_logfactor=float(current_explicit[0]),
                absolute_difference_nat=default_identity_error,
                tolerance_nat=2e-6, support=current_support),
            count_conditioned=dict(normalized_raw_FP_logfactor=count_value,
                source_radius_cMpc_h=point_radius, source_mixture=count_mixture,
                active_components=len(ids), support_at_logscale_offsets=info_by_scale,
                support_superset_verified=True, LOS_logscale_AD=count_ad,
                LOS_logscale_FD=count_fd, AD_FD_relative_error=count_error),
            independent_periodic_ray=dict(normalized_raw_FP_logfactor=ray_value,
                source_radius_cMpc_h=group_radius, source_mixture=ray_mixture,
                first_face_radius_cMpc_h=face,
                beyond_first_face_radial_weight_fraction=ray_image_fraction,
                LOS_logscale_AD=ray_ad, LOS_logscale_FD=ray_fd,
                AD_FD_relative_error=ray_error,
                interpretation='fixed-state comparator, not a count-conditioned likelihood'),
            comparison=dict(count_conditioned_minus_independent_ray_logfactor_nat=
                count_value-ray_value,
                noncentral_voxel_image_geometric_margin_cMpc_h=image_margin,
                voxel_cdf_image_boxes_evaluated=27),
            input_sha256={str(p): sha256(p) for p in
                (STATE, SPLIT, SOURCE, FP, ROOT/'data/cf4_2mpp_crossmatch_v1.csv')})
        save()
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='FAILED_COUNT_FP_OWNERSHIP_CONTROL', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
