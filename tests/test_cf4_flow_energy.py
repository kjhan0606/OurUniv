import unittest

import numpy as np
import torch

from cf4_flow_energy import (energy_coefficients, sample_trace, trace_backward,
                             device_record, normalize_groups)
from cf4_conditional_split_flow import ConditionalSplitFlow
from cf4_continuous_matter import restrict


class EnergyTests(unittest.TestCase):
    def test_mixed_atom_continuous_energy_gradient(self):
        # Exact atom plus Gauss-Legendre integration of Beta(a,1) on fixed(0,1).
        # Both mixture probability and continuous shape have nonzero gradients.
        points, weights = np.polynomial.legendre.leggauss(160)
        x = torch.tensor(np.r_[0., (points+1)/2], dtype=torch.float64)
        measure = torch.tensor(np.r_[1., weights/2], dtype=torch.float64)
        theta = torch.tensor([.3, np.log(2.)], dtype=torch.float64, requires_grad=True)
        p, a = theta[0].sigmoid(), theta[1].exp()
        density = torch.cat(((1-p)[None], p*a*x[1:]**(a-1)))
        probs = density*measure
        target = .43
        distance = abs(x-target)
        between = abs(x[:, None]-x[None, :])
        es = (distance[:, None]+distance[None, :]-between)/2
        expected = (probs[:, None]*probs[None, :]*es).sum()
        exact = torch.autograd.grad(expected, theta, retain_graph=True)[0]
        score = torch.stack([torch.autograd.grad(v, theta, retain_graph=True)[0]
                             for v in density.log()])
        first = (distance[:, None]-between)/2
        second = (distance[None, :]-between)/2
        estimated = (probs[:, None, None]*probs[None, :, None]*
                     (first[:, :, None]*score[:, None]+second[:, :, None]*score[None])).sum((0, 1))
        torch.testing.assert_close(exact, estimated, atol=2e-8, rtol=2e-7)
        self.assertTrue(torch.all(abs(exact) > 1e-3))
        value, multipliers = energy_coefficients(np.array([.2]), np.array([.8]), np.array([target]))
        self.assertAlmostEqual(value, .5*(.23+.37-.6))
        np.testing.assert_allclose(multipliers, [.5*(.23-.6), .5*(.37-.6)])
        # A fixed past-sample baseline integrates to zero, not a same-pair baseline.
        centered = estimated - .7*(probs[:, None]*score).sum(0)
        torch.testing.assert_close(exact, centered, atol=3e-8, rtol=3e-7)

    def test_complete_trace_gradient_and_conservation(self):
        torch.manual_seed(173)
        model = ConditionalSplitFlow(np.zeros(7), np.ones(7))
        root = np.array([10., 100., 200., 300., 1400., 4400., 9400.]).reshape(7, 1, 1, 1)
        field, trace = sample_trace(model, root, [.375, .1875])
        self.assertEqual(len(trace), 14)
        np.testing.assert_allclose(restrict(field, 4), root, rtol=1e-12, atol=1e-10)
        model.zero_grad(set_to_none=True)
        value = trace_backward(model, trace, .37)
        summed = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        monolithic = sum(model.log_prob(*device_record(*r, 'cpu')).sum() for r in trace)
        np.testing.assert_allclose(value, float(monolithic.detach()), atol=1e-4, rtol=2e-6)
        (.37*monolithic).backward()
        for parameter, reference in zip(model.parameters(), summed):
            torch.testing.assert_close(parameter.grad, reference, atol=2e-5, rtol=2e-5)
        model.zero_grad(set_to_none=True)
        trace_backward(model, trace[7:], .37)
        self.assertGreater(sum(float((p.grad-r).abs().sum()) for p, r in zip(model.parameters(), summed)), 1e-3)

    def test_fixed_feature_scales_and_unavailable_group(self):
        np.testing.assert_allclose(normalize_groups([np.array([2.]), np.array([9.]), np.array([6.])], [2., 0., 3.]),
                                   np.array([1., 2.])/np.sqrt(2))
        with self.assertRaises(ValueError):
            normalize_groups([np.ones(1)], [0.])


if __name__ == '__main__':
    unittest.main()
