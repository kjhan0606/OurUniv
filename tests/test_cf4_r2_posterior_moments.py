import unittest
import numpy as np
from cf4_r2_posterior_moments import PresentMomentAccumulator


def field(rho,velocity,variance):
    rho=np.asarray(rho,dtype=float).reshape(2,2,2)
    v=np.broadcast_to(np.asarray(velocity,dtype=float),rho.shape).copy()
    return dict(rho=rho,mean_velocity_km_s=np.stack([v,v*2,v*3]),
        physical_velocity_variance_km2_s2=np.full((3,2,2,2),variance),velocity_valid=rho>0)


class MomentTests(unittest.TestCase):
    def test_rejections_are_repeated_samples_and_dispersion_is_separate(self):
        a=field(np.ones(8),10.,4.);b=field(np.full(8,4.),40.,9.)
        acc=PresentMomentAccumulator((2,2,2))
        for f in (a,a,b):acc.update(f)
        out=acc.arrays()
        np.testing.assert_allclose(out['rho_mean'],2.)
        np.testing.assert_allclose(out['rho_posterior_variance'],3.)
        np.testing.assert_allclose(out['conditional_mean_velocity_km_s'][0],20.)
        np.testing.assert_allclose(out['conditional_velocity_posterior_variance_km2_s2'][0],300.)
        np.testing.assert_allclose(out['conditional_mean_physical_velocity_variance_km2_s2'],17/3)
        self.assertEqual(int(out['retained_states']),3)
        self.assertFalse(bool(out['posterior_MC_error_estimated']))

    def test_empty_cells_are_not_zero_velocity_observations(self):
        a=field([0,0,1,1,1,1,1,1],0.,0.)
        b=field([1,0,1,1,1,1,1,1],30.,9.)
        acc=PresentMomentAccumulator((2,2,2));acc.update(a);acc.update(b)
        out=acc.arrays()
        self.assertEqual(out['conditional_mean_velocity_km_s'][0,0,0,0],30.)
        self.assertTrue(np.isnan(out['conditional_velocity_posterior_variance_km2_s2'][0,0,0,0]))
        self.assertTrue(np.isnan(out['conditional_mean_velocity_km_s'][0,0,0,1]))
        self.assertEqual(out['occupancy_fraction'][0,0,0],.5)

    def test_resume_matches_uninterrupted_and_invalid_update_does_not_mutate(self):
        a=field(np.ones(8),10.,4.);b=field(np.full(8,4.),40.,9.)
        acc=PresentMomentAccumulator((2,2,2));acc.update(a)
        resumed=PresentMomentAccumulator.from_checkpoint(acc.checkpoint_arrays())
        acc.update(b);resumed.update(b)
        for k,v in acc.arrays().items():np.testing.assert_allclose(v,resumed.arrays()[k],equal_nan=True)
        bad=dict(a,velocity_valid=np.zeros((2,2,2),dtype=bool))
        with self.assertRaises(ValueError):resumed.update(bad)
        self.assertEqual(resumed.n,2)


if __name__=='__main__':unittest.main()
