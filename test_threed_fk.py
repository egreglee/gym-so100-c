import unittest
import numpy as np
import math
import threed_fk as d3

class TestThreedFk(unittest.TestCase):
    def test_zero(self):
        zeros = np.zeros((6,))
        want = [0.02027, 0.5, 0.1039]
        got = d3.threed_fk(zeros, False)
        got_mjg = got['moving_jaw_grasp']
        for i, w in enumerate(want):
            self.assertAlmostEqual(got_mjg[i], w, delta=5e-7)

    def test_wristup(self):
        wristup = [-6.57574296e-11, -2.27688006e-01, -3.80616735e-07,  1.44596769e-06, 0, 0]
        want = [0.0130072, 0.50000000, 0.19762045]
        got = d3.threed_fk(wristup, False)
        got_mjg = got['moving_jaw_grasp']
        for i, w in enumerate(want):
            self.assertAlmostEqual(got_mjg[i], w, delta=5e-7)

    def test_general(self):
        # especially non-zero rotation
        general = [ 1.6704086,  -0.72340645,  0.69616856, -1.09520214,  1.59030296, -0.15477658]
        want =  [-0.4528063,   0.837627192,  0.32841733]
        got = d3.threed_fk(general, False)
        got_mjg = got['moving_jaw_grasp']
        for i, w in enumerate(want):
            self.assertAlmostEqual(got_mjg[i], w, delta=5e-7)
        
    def test_rotx(self):
        xyz = np.array([1,2,3])
        q = np.pi/2
        want = np.array([1.0, -3.0, 2.0])
        got = d3.rotx(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

    def test_rotx_pi4(self):
        xyz = np.array([1,2,3])
        q = np.pi/4
        want = np.array([1, -1/math.sqrt(2), 5/math.sqrt(2)])
        got = d3.rotx(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

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
        want = np.array([-2.0, 1.0, 3.0])
        got = d3.rotz(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

    def test_rotz_pi4(self):
        xyz = np.array([1,2,3])
        q = np.pi/4
        want = np.array([-0.70710678, 2.12132034, 3.])
        got = d3.rotz(q, xyz)
        for i in range(len(got)):
            self.assertAlmostEqual(got[i], want[i], delta=5e-7)

if __name__ == "__main__":
    unittest.main()
