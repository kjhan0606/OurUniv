import unittest

import numpy as np

from cf4_r2_lowk_geometry_attribution import (
    analyze_spectra, fourier_geometry, multipole_projection)


class LowKGeometryAttributionTest(unittest.TestCase):
    def test_shell_radial_sum_matches_parseval_and_fundamental_pair(self):
        n = 32
        rng = np.random.default_rng(20261002)
        q = rng.normal(size=(n,n,n))
        nll = rng.normal(size=(n,n,n))
        q_hat = np.fft.fftn(q,norm='ortho')
        nll_hat = np.fft.fftn(nll,norm='ortho')
        result = analyze_spectra(q_hat,q_hat+nll_hat,max_k_index=8)

        np.testing.assert_allclose(result['all_mode_radial']['likelihood_nll'],
            np.sum(q*nll),rtol=0.,atol=1e-10)
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
        kx,ky,kz=np.meshgrid(axis,axis,axis,indexing='ij')
        lowk=(kx*kx+ky*ky+kz*kz>0)&(kx*kx+ky*ky+kz*kz<=64)
        direct_lowk=np.sum(np.real(np.conj(q_hat[lowk])*nll_hat[lowk]))
        np.testing.assert_allclose(result['lowk_shells']['likelihood_nll_radial'],
            direct_lowk,rtol=0.,atol=1e-10)
        mode = (1,0,0)
        expected = 2*np.real(np.conj(q_hat[mode])*nll_hat[mode])
        row = next(row for row in result['fundamental_mode_checks'] if tuple(row['mode'])==mode)
        np.testing.assert_allclose(row['radial_nll_derivative_pair'],expected,rtol=0.,atol=1e-12)

    def test_centered_monopole_phase_isolated(self):
        n = 16
        *_, shell_squared, phase, directions = fourier_geometry(n,4)
        mode = shell_squared==1
        kx,ky,kz,kmag,_,_,_ = fourier_geometry(n,4)
        profile = np.exp(-.2*kmag[mode]**2)
        force = phase[mode]*profile
        result = multipole_projection(force,directions[mode],phase[mode])

        self.assertGreater(result['component_fraction']['monopole_l0_fraction'],1-1e-12)
        self.assertLess(result['component_fraction']['dipole_l1_fraction'],1e-12)
        self.assertLess(result['component_fraction']['quadrupole_l2_fraction'],1e-12)
        self.assertLess(result['component_fraction']['higher_order_or_unmodelled_fraction'],1e-12)
        self.assertGreater(result['centered_monopole_coefficient'],0.)

    def test_monopole_dipole_and_quadrupole_projection_reconstructs_known_band(self):
        n = 16
        kx,ky,kz,kmag,shell_squared,phase,directions = fourier_geometry(n,4)
        mode = shell_squared==5
        direction = directions[mode].astype(np.float64)
        x,y,z=direction.T
        quad_basis=np.column_stack((x*x-y*y,x*x-z*z,2*x*y,2*y*z,2*z*x))
        dipole_coef=np.array([.3,-.2,.1])
        quad_coef=np.array([.15,-.1,.2,-.08,.11])
        corrected=.7+1j*(direction@dipole_coef)+quad_basis@quad_coef
        force=phase[mode]*corrected
        result=multipole_projection(force,direction,phase[mode])

        self.assertGreater(result['component_fraction']['monopole_l0_fraction'],0.)
        self.assertGreater(result['component_fraction']['dipole_l1_fraction'],0.)
        self.assertGreater(result['component_fraction']['quadrupole_l2_fraction'],0.)
        self.assertLess(result['component_fraction']['higher_order_or_unmodelled_fraction'],1e-24)
        np.testing.assert_allclose(result['dipole_coefficients'],dipole_coef,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(result['quadrupole_coefficients'],quad_coef,rtol=0.,atol=1e-12)


if __name__=='__main__':
    unittest.main()
