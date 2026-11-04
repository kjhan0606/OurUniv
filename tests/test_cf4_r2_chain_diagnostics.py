import unittest
import numpy as np
from cf4_r2_chain_diagnostics import (retained_field_observables,
    four_chain_scalar_diagnostics,scalar_diagnostics,retained_scalar_matrix)


class DiagnosticsTests(unittest.TestCase):
    @staticmethod
    def synthetic_chain_report(chain_index):
        rng=np.random.default_rng(100+chain_index)
        trace=[dict(iteration=index+1,warmup=True) for index in range(8)]
        for index in range(8):
            row=dict(warmup=False,
                iteration=index+9,
                fine_energy=float(rng.normal()),
                white_mean_square=float(rng.normal()),
                inherited_low_white_mean_square=float(rng.normal()),
                nuisance_white=rng.normal(size=24).tolist(),
                fundamental_real=rng.normal(size=3).tolist(),
                fundamental_imag=rng.normal(size=3).tolist(),
                count_raw_scores=rng.normal(size=2).tolist(),
                density_octant_means=rng.normal(size=8).tolist(),
                density_mode_real=rng.normal(size=3).tolist(),
                density_mode_imag=rng.normal(size=3).tolist(),
                density_band_power_k2_1=float(rng.uniform(.1,2.)),
                density_band_power_k2_4_8=float(rng.uniform(.1,2.)),
                momentum_mode_real=rng.normal(size=9).tolist(),
                momentum_mode_imag=rng.normal(size=9).tolist(),
                training_region_density_mean=float(rng.uniform(.5,1.5)),
                training_region_occupancy_fraction=float(rng.uniform(.1,1.)),
                training_region_mass_weighted_velocity_km_s=rng.normal(size=3).tolist())
            trace.append(row)
        streams={name:dict(stream_id=stream_id,
            spawn_key=[chain_index,stream_id])
            for name,stream_id in (('prior',0),('audit',1),('transition',2))}
        return dict(status='N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS',
            R2_complete=False,heldout_scored=False,long_chain=True,
            initialization_policy='independent_prior',
            chain_rng_design=dict(master_seed=3,chain_index=chain_index,
                chain_count=4,bit_generator='PCG64',streams=streams),
            chain_id=f'chain-{chain_index}',source_commit='commit',
            target='conditional-v6',N=256,box_cMpc_h=384.,
            fine_volume_order=2,force_volume_order=1,
            max_proposals=16,warmup_proposals=8,
            schedule=dict(proposals=16,warmup=8),
            field_mcse_contract=dict(batch_sizes=[2,4],minimum_batches=2,
                retained_capacity=8),
            association_ledger_source_commit='association-commit',
            observation_target_revision='conditional-v6-test',
            source_conditioning_rule='test-only',
            prior_start_selection=dict(status='FINITE_PRIOR_START_SELECTED',
                max_attempts=4,attempts=[dict(attempt=1,status='FINITE',
                    state_sha256=f'{chain_index+1:064x}')],selected_attempt=1,
                selected_state_sha256=f'{chain_index+1:064x}'),
            retained_states=8,retained_state_records=8,trace=trace)

    def test_iid_and_displaced_chains(self):
        x=np.random.default_rng(91).normal(size=(2,10000))
        d=scalar_diagnostics(x)
        self.assertLess(d['rank_folded_split_rhat'],1.01)
        self.assertGreater(d['batch_means_ess'],10000)
        x[1]+=3
        self.assertGreater(scalar_diagnostics(x)['rank_folded_split_rhat'],1.2)

    def test_correlated_and_stuck_chains(self):
        x=np.zeros((2,10000));noise=np.random.default_rng(4).normal(size=x.shape)
        for i in range(1,x.shape[1]):x[:,i]=.95*x[:,i-1]+noise[:,i]
        self.assertLess(scalar_diagnostics(x)['batch_means_ess'],2000)
        d=scalar_diagnostics(np.ones((2,20)))
        self.assertIsNone(d['rank_folded_split_rhat']);self.assertIsNone(d['batch_means_ess'])
        self.assertIsNone(d['bulk_ess']);self.assertIsNone(d['tail_ess'])
        with self.assertRaises(ValueError):scalar_diagnostics(np.ones((2,3)))

    def test_rank_bulk_tail_ess_for_iid_and_ar1_chains(self):
        n=1200
        rng=np.random.default_rng(114)
        iid=rng.normal(size=(4,n))
        iid_diagnostics=scalar_diagnostics(iid)
        self.assertLess(iid_diagnostics['rank_folded_split_rhat'],1.01)
        self.assertGreater(iid_diagnostics['bulk_ess'],.75*4*n)
        self.assertGreater(iid_diagnostics['tail_ess'],.65*4*n)

        phi=.8
        noise=np.random.default_rng(115).normal(size=(4,n))
        ar1=np.empty_like(noise)
        ar1[:,0]=noise[:,0]/np.sqrt(1.-phi**2)
        for index in range(1,n):
            ar1[:,index]=phi*ar1[:,index-1]+noise[:,index]
        diagnostics=scalar_diagnostics(ar1)
        theoretical=4*n*(1.-phi)/(1.+phi)
        self.assertLess(diagnostics['bulk_ess'],.5*4*n)
        self.assertGreater(diagnostics['bulk_ess'],.5*theoretical)
        self.assertLess(diagnostics['bulk_ess'],1.5*theoretical)
        self.assertIsNotNone(diagnostics['tail_ess'])
        self.assertIn('initial monotone sequence',diagnostics['ess_method'])

    def test_unequal_chain_lengths_are_rejected_not_trimmed(self):
        chains=[np.arange(32.),np.arange(31.)]
        with self.assertRaisesRegex(ValueError,'equal-length'):
            scalar_diagnostics(chains)

    def test_retained_field_modes_and_training_region_are_explicit(self):
        n=4
        x=np.arange(n)[:,None,None]
        rho=np.broadcast_to(1.+.5*np.cos(2*np.pi*x/n),(n,n,n)).copy()
        velocity=np.zeros((3,n,n,n));velocity[0]=2.
        field=dict(rho=rho,mean_velocity_km_s=velocity,
            velocity_valid=np.ones((n,n,n),dtype=bool))
        result=retained_field_observables(field,np.ones((n,n,n),dtype=bool))
        self.assertAlmostEqual(result['density_mode_real'][0],2.,places=12)
        self.assertAlmostEqual(result['density_mode_imag'][0],0.,places=12)
        self.assertAlmostEqual(result['density_band_power_k2_1'],8./6.,places=12)
        self.assertAlmostEqual(result['momentum_mode_real'][0],4.,places=12)
        self.assertEqual(result['training_region_density_mean'],1.)
        self.assertEqual(result['training_region_occupancy_fraction'],1.)
        self.assertEqual(result['training_region_mass_weighted_velocity_km_s'],[2.,0.,0.])

    def test_field_observable_trace_is_required_when_requested(self):
        row=dict(warmup=False,fine_energy=1.,white_mean_square=1.,
            inherited_low_white_mean_square=.7,nuisance_white=[0.]*24,
            fundamental_real=[0.]*3,fundamental_imag=[0.]*3,
            count_raw_scores=[0.]*2,density_octant_means=[1.]*8)
        with self.assertRaisesRegex(ValueError,'lacks required'):
            retained_scalar_matrix({'trace':[row]},require_field_observables=True)
        field=dict(rho=np.ones((4,4,4)),
            mean_velocity_km_s=np.ones((3,4,4,4)),
            velocity_valid=np.ones((4,4,4),dtype=bool))
        row.update(retained_field_observables(field,np.ones((4,4,4),dtype=bool)))
        labels,values=retained_scalar_matrix({'trace':[row]},
            require_field_observables=True)
        self.assertIn('density_mode_real_0',labels)
        self.assertIn('momentum_mode_imag_8',labels)
        self.assertEqual(values.shape,(1,len(labels)))

    def test_four_chain_gate_is_fail_closed_and_never_promotes(self):
        reports=[self.synthetic_chain_report(index) for index in range(4)]
        result=four_chain_scalar_diagnostics(reports)
        self.assertEqual(result['n_chains'],4)
        self.assertEqual(result['retained_per_chain'],[8]*4)
        self.assertEqual(result['field_gate'],'NOT_EVALUATED')
        self.assertFalse(result['R2_complete'])
        self.assertFalse(result['heldout_scored'])
        self.assertEqual(result['status'],
            'FOUR_CHAIN_SCALAR_DIAGNOSTICS_ONLY_FIELD_GATE_PENDING')
        self.assertEqual(result['scalar_gate'],'FAIL')

    def test_four_chain_gate_rejects_missing_chain_and_heldout_access(self):
        reports=[self.synthetic_chain_report(index) for index in range(4)]
        with self.assertRaisesRegex(ValueError,'exactly four'):
            four_chain_scalar_diagnostics(reports[:3])
        reports[2]['heldout_scored']=True
        with self.assertRaisesRegex(ValueError,'pre-heldout'):
            four_chain_scalar_diagnostics(reports)

    def test_four_chain_gate_checks_stream_identity_and_complete_trace(self):
        reports=[self.synthetic_chain_report(index) for index in range(4)]
        reports[1]['chain_rng_design']['streams']['prior']['spawn_key']=[0,0]
        with self.assertRaisesRegex(ValueError,'stream map'):
            four_chain_scalar_diagnostics(reports)
        reports=[self.synthetic_chain_report(index) for index in range(4)]
        reports[3]['trace'].pop()
        with self.assertRaisesRegex(ValueError,'trace is incomplete'):
            four_chain_scalar_diagnostics(reports)

    def test_four_chain_gate_requires_distinct_well_formed_prior_starts(self):
        reports=[self.synthetic_chain_report(index) for index in range(4)]
        reports[3]['prior_start_selection']['selected_state_sha256'] = \
            reports[2]['prior_start_selection']['selected_state_sha256']
        reports[3]['prior_start_selection']['attempts'][0]['state_sha256'] = \
            reports[2]['prior_start_selection']['selected_state_sha256']
        with self.assertRaisesRegex(ValueError,'not distinct'):
            four_chain_scalar_diagnostics(reports)

        reports=[self.synthetic_chain_report(index) for index in range(4)]
        reports[1]['prior_start_selection']['attempts'][0]['status'] = \
            'NUMERICAL_OR_TARGET_FAILURE'
        with self.assertRaisesRegex(ValueError,'invalid retry'):
            four_chain_scalar_diagnostics(reports)

    def test_rejections_remain_in_retained_trace(self):
        row=dict(warmup=False,accepted=False,fine_energy=1.,white_mean_square=1.,
            inherited_low_white_mean_square=.7,nuisance_white=[0.]*24,
            fundamental_real=[0.]*3,fundamental_imag=[0.]*3,count_raw_scores=[0.]*2,
            density_octant_means=[1.]*8)
        labels,values=retained_scalar_matrix({'trace':[dict(row,warmup=True),row,row]})
        self.assertEqual(values.shape,(2,len(labels)))
        np.testing.assert_array_equal(values[0],values[1])


if __name__=='__main__':unittest.main()
