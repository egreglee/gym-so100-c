import math
import numpy as np
import numpy as np
import threed_fk as d3

def get_transform(theta, d, a, alpha):
    """Standard DH Matrix: Handles axis swaps and sign changes automatically."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

# [a (length), d (offset), alpha (twist), theta_offset]
# Derived from your site_xpos at q=0
dh_params_ = [
    [0.0,     0.0165,  np.pi/2,  np.pi/2], # Base Euler + Rotation
    [0.1025,  0.0452,  np.pi/2,  0.0],     # Pitch
    [0.11257, 0.0306,  0.0,      0.0],     # Elbow
    [0.1349,  0.028,  -np.pi/2,  0.0]      # Wrist Pitch
]

# [a (length), d (offset), alpha (twist), theta_offset]
dh_params = [
    [0.0452,  0.0165,  np.pi/2,  np.pi/2], # 1. Base to Rotation Pivot
    [0.1025,  0.0306,  0.0,      0.0],     # 2. Rotation to Pitch
    [0.11257, 0.028,   0.0,      0.0],     # 3. Pitch to Elbow
    [0.1349,  0.0052,  -np.pi/2, 0.0]      # 4. Elbow to Wrist
]
def forward_kinematics_v3(joint_angles):
    # Start at World Base [-0.469, 0.5, 0]
    T = np.eye(4)
    T[:3, 3] = [-0.469, 0.5, 0]
    
    for i, q in enumerate(joint_angles):
        if i >= len(dh_params): break
        a, d, alpha, q_off = dh_params[i]
        T = T @ get_transform(q + q_off, d, a, alpha)
        with np.printoptions(precision=5):
            print(f"{T}")
            pass
    return T

def joint_vec2d(q, L, O):
    xz = np.array([L, O])
    (c, s) = np.cos(q), np.sin(q)
    return np.array(((c, -s), (s, c))) @ xz
    
def planar_fk_v3(q_raw, verbose):

    q = q_raw.copy()
    for i in range(1,4):
        q[i] = -q[i] # reverse pitch, elbow, wrist_pitch rotation convention

    L_rotate, O_rotate = 0.0452, 0.0165           # base to rotate
    L_pitch, O_pitch =  0.0306, 0.1025            # rotate to pitch
    L_elbow, O_elbow =  0.11257, -0.028           # pitch to elbow
    L_wrist_pitch, O_wrist_pitch = 0.1349,0.0052  # elbow to wrist_pitch
    L_wrist_roll, O_wrist_roll = 0.0601, 0        # wrist_pitch to wrist_roll
    L_jaw_target, O_jaw_target = 0.10125, 0       # wrist_roll to jaw_target

    # compute net reach vector from rotate joint to jaw_target in planar projection
    xz = np.zeros((2,))

    joint = "pitch"
    xz_q0 = np.array([L_pitch, O_pitch])
    xz += xz_q0
    if verbose:
        print(f"{joint:11} {xz} = {xz_q0}")
    
    joint = "elbow"
    xz_q1 = joint_vec2d(q[1], L_elbow, O_elbow)
    xz_prev = xz
    xz += xz_q1
    if verbose:
        print(f"{joint:11} {xz} = {xz_prev} + {xz_q1}")

    joint = "wrist_pitch"
    xz_q2 = joint_vec2d(q[1] + q[2], L_wrist_pitch, O_wrist_pitch)
    xz_prev = xz
    xz += xz_q2
    if verbose:
        print(f"{joint:11} {xz} = {xz_prev} + {xz_q2}")
    
    joint = "wrist_roll"
    xz_q3 = joint_vec2d(q[1] + q[2] + q[3], L_wrist_roll, O_wrist_roll)
    xz_prev = xz
    xz += xz_q3
    if verbose:
        print(f"{joint:11}  {xz} = {xz_prev} + {xz_q3}")

    joint = "jaw_target"
    xz_jt = joint_vec2d(q[1] + q[2] + q[3] + 0, L_jaw_target, O_jaw_target)
    xz_prev = xz
    xz += xz_jt
    if verbose:
        print(f"{joint:11}  {xz} = {xz_prev} + {xz_jt}")
    
    # rotate vector in xy
    (c, s) = np.cos(q[0]), np.sin(q[0])
    x_0 = np.array((xz[0], 0))
    xy = np.array(((c, -s), (s, c))) @ x_0
    if verbose:
        print(f"xy {xy}")

    # world coordinates of rotate joint from sim
    x_rotate, y_rotate, z_rotate = -0.4238, 0.50000017, 0.0165
    world_x = x_rotate + xy[0]
    world_y = y_rotate + xy[1]
    world_z = z_rotate + xz[1]
    return np.array([world_x, world_y, world_z])

def main():
    forward = forward_kinematics_v3([0,0,0,0,0,0])
    #print(f"forward {forward}")
    # Example: All joints at 0
    zeros = np.zeros((6,))
    print(f"Planar Prediction v3: {planar_fk_v3(zeros, True)}")
    wristup = np.array([-6.57574296e-11, -2.27688006e-01, -3.80616735e-07,  1.44596769e-06, 0, 0])
    print(f"Planar Prediction v3: {planar_fk_v3(wristup, True)} @ {wristup}")

    print(f"--------------------")
    zeros = np.zeros((6,))
    print(f"threed_fk: {d3.threed_fk(zeros, True)}")
    print(f"threed_fk: {d3.threed_fk(wristup, True)} @ {wristup}")

if __name__ == "__main__":
    main()
