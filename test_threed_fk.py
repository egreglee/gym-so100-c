import unittest
import numpy as np
import threed_fk as d3

class TestThreedFk(unittest.TestCase):
    def test_zero(self):
        zeros = np.zeros((6,))
        want = [0.01562, 0.50000017, 0.0962]
        got = d3.threed_fk(zeros, False)
        got_jt = got['jaw_target']
        for i, w in enumerate(want):
            self.assertAlmostEqual(got_jt[i], w, delta=5e-7)

    def test_wristup(self):
        wristup = [-6.57574296e-11, -2.27688006e-01, -3.80616735e-07,  1.44596769e-06]
        want = [0.01021529, 0.50000017, 0.18906955]
        got = d3.threed_fk(wristup, False)
        got_jt = got['jaw_target']
        for i, w in enumerate(want):
            self.assertAlmostEqual(got_jt[i], w, delta=5e-7)

    def test_roty(self):
        xyz = np.array([1,2,3])
        q = np.pi/2
        want = np.array([3.0, 2.0, -1.0])
        got = d3.roty(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

    def test_roty_pi4(self):
        xyz = np.array([1,2,3])
        q = np.pi/4
        want = np.array([2.82842712, 2., 1.41421356])
        got = d3.roty(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

    def test_rotz(self):
        xyz = np.array([1,2,3])
        q = np.pi/2
        want = np.array([2.0, -1.0, 3.0])
        got = d3.rotz(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

    def test_rotz_pi4(self):
        xyz = np.array([1,2,3])
        q = np.pi/4
        want = np.array([2.12132034, 0.70710678, 3.])
        got = d3.rotz(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

if __name__ == "__main__":
    unittest.main()
