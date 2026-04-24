import numpy as np

def joint_vec2d(q, L, O):
    xz = np.array([L, O])
    (c, s) = np.cos(q), np.sin(q)
    return np.array(((c, -s), (s, c))) @ xz

def rotx(q, xyz):
    (c, s) = np.cos(q), np.sin(q)
    rot = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]])
    return rot @ xyz

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
    L_jaw_grasp, O_jaw_grasp = 0.10250, 0        # wrist_roll to jaw_grasp

    # jaw_grasp_site pos = "0 -0.10250 0"
    # body MOVING_JAW relative to body FIXED_JAW
    jaw_site = [-0.0202, -0.0244, 0]

    # sites marking grasp points of jaw
    # relative to body MOVING_JAW (joint JAW) 
    moving_jaw_grasp_site = [-0.0125, -0.0815, 0]
    # relative to body FIXED_JAW (joint WRIST_ROLL)
    jaw_grasp_site = [np.nan, -0.10250, np.nan]
    fixed_jaw_grasp_site =  [0.0075, -0.10250, 0]

    location = 'base'
    xyz = base
    xyz_prev = xyz.copy()
    ret[location] = xyz.copy()
    
    location = "rotate"
    xyz += [L_rotate, 0, O_rotate]
    ret[location] = xyz.copy()
    
    location = "pitch"
    xyz += rotz(q[0], [L_pitch, 0, O_pitch])
    ret[location] = xyz.copy()
    
    location = "elbow"
    delta = rotz(q[0], roty(q[1], [L_elbow, 0, O_elbow]))
    xyz_prev = xyz.copy()
    xyz += delta
    ret[location] = xyz.copy()

    location = "wrist_pitch"
    delta = rotz(q[0], roty(q[1] + q[2], [L_wrist_pitch, 0, O_wrist_pitch]))
    xyz_prev = xyz.copy()
    xyz += delta
    ret[location] = xyz.copy()
    
    location = "wrist_roll"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], [L_wrist_roll, 0, O_wrist_roll]))
    xyz_prev = xyz.copy()
    xyz += delta
    if verbose :
        print(f"{location:16}  {xyz} = {xyz_prev} + {delta}")
    ret[location] = xyz.copy()

    xyz_wrist_roll = xyz.copy()
    
    location = "jaw"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], rotx(-q[4], [-jaw_site[1], jaw_site[2], -jaw_site[0]])))
    xyz = xyz_wrist_roll + delta
    ret[location] = xyz.copy()
    xyz_jaw = xyz.copy()
    if verbose :
        print(f"{location:16}  {xyz}")

    location = "moving_jaw_grasp"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], rotx(-q[4], roty(-q[5], [-moving_jaw_grasp_site[1], moving_jaw_grasp_site[2], moving_jaw_grasp_site[0]]))))
    xyz = xyz_jaw + delta
    ret[location] = xyz.copy()
    if verbose :
        print(f"{location:16}  {xyz} = {xyz_jaw} + {delta}")

    location = "fixed_jaw_grasp"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], rotx(-q[4], [-fixed_jaw_grasp_site[1], fixed_jaw_grasp_site[2], -fixed_jaw_grasp_site[0]])))
    xyz = xyz_wrist_roll + delta
    ret[location] = xyz.copy()

    location = "jaw_grasp"
    delta = rotz(q[0], roty(q[1] + q[2] + q[3], [-jaw_grasp_site[1], 0, 0]))
    xyz = xyz_wrist_roll + delta
    ret[location] = xyz.copy()

    return ret

