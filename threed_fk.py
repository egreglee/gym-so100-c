import numpy as np

def joint_vec2d(q, L, O):
    xz = np.array([L, O])
    (c, s) = np.cos(q), np.sin(q)
    return np.array(((c, -s), (s, c))) @ xz

def roty(q, xyz):
    (c, s) = np.cos(q), np.sin(q)
    rot = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]])
    return rot @ xyz

def rotz(q, xyz):
    (c, s) = np.cos(q), np.sin(q)
    rot = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])
    return rot @ xyz

def xz_x0z(xz):
    return np.array([xz[0], 0, xz[1]])

def threed_fk(q, verbose):
    ret = {}

    base = np.array([-0.469, 0.5, 0])
    L_rotate, O_rotate = 0.0452, 0.0165           # base to rotate
    L_pitch, O_pitch =  0.0306, 0.1025            # rotate to pitch
    L_elbow, O_elbow =  0.11257, -0.028           # pitch to elbow
    L_wrist_pitch, O_wrist_pitch = 0.1349, 0.0052 # elbow to wrist_pitch
    L_wrist_roll, O_wrist_roll = 0.0601, 0        # wrist_pitch to wrist_roll
    L_jaw_grasp, O_jaw_grasp = 0.10125, 0        # wrist_roll to jaw_grasp

    # body MOVING_JAW relative to body FIXED_JAW
    jaw_site = [-0.0202, -0.0244, 0]

    # sites marking grasp points of jaw
    # relative to body MOVING_JAW (joint JAW) 
    moving_jaw_grasp_site = [-0.0125, -0.0765, 0]
    # relative to body FIXED_JAW (joint WRIST_ROLL)
    jaw_grasp_site = [0, -0.10125, 0]
    fixed_jaw_grasp_site =  [0.0075, -0.10125, 0]

    joint = 'base'
    xyz = base
    xyz_prev = xyz.copy()
    ret[joint] = xyz.copy()
    
    joint = "rotate"
    xyz += [L_rotate, 0, O_rotate]
    ret[joint] = xyz.copy()
    
    joint = "pitch"
    xyz += rotz(q[0], [L_pitch, 0, O_pitch])
    if verbose:
        print(f"{joint:11} {xyz} = {xyz_q0}")
    ret[joint] = xyz.copy()
    
    joint = "elbow"
    delta = rotz(q[0], roty(q[1], [L_elbow, 0, O_elbow]))
    xyz_prev = xyz.copy()
    xyz += delta
    if verbose :
        print(f"{joint:11} {xyz} = {xyz_prev} + {delta}")
    ret[joint] = xyz.copy()

    joint = "wrist_pitch"
    delta = rotz(q[0], roty(q[1] + q[2], [L_wrist_pitch, 0, O_wrist_pitch]))
    xyz_prev = xyz.copy()
    xyz += delta
    if verbose :
        print(f"{joint:11} {xyz} = {xyz_prev} + {delta}")
    ret[joint] = xyz.copy()
    
    joint = "wrist_roll"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], [L_wrist_roll, 0, O_wrist_roll]))
    xyz_prev = xyz.copy()
    xyz += delta
    if verbose :
        print(f"{joint:11}  {xyz} = {xyz_prev} + {delta}")
    ret[joint] = xyz.copy()

    xyz_wrist_roll = xyz.copy()
    
    joint = "fixed_jaw_grasp"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], [L_jaw_grasp, 0, O_jaw_grasp]))
    xyz = xyz_wrist_roll + delta
    if verbose :
        print(f"{joint:11}  {xyz} = {xyz_prev} + {delta}")
    ret[joint] = xyz.copy()

    joint = "jaw_grasp"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], [L_jaw_grasp, 0, O_jaw_grasp]))
    xyz = xyz_wrist_roll + delta
    if verbose :
        print(f"{joint:11}  {xyz} = {xyz_prev} + {delta}")
    ret[joint] = xyz.copy()

    return ret

