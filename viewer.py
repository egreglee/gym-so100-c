import argparse
import numpy as np
import mujoco
import mujoco.viewer
import dh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-g", action="store_true", help="simulate with no gravity")
    args = parser.parse_args()
    print(f"--zero-g {args.zero_g}")
    model = mujoco.MjModel.from_xml_path('/home/greg/gym-so100-c/gym_so100/assets/so100_transfer_cube.xml')
    data = mujoco.MjData(model)
    if args.zero_g:
        model.opt.gravity[:] = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        site_prev = np.zeros((3,))
        settled = False  # for convergence
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # monitor jaw target
            site_name = "jaw_target_site"
            site_world_pos = data.site(site_name).xpos
            # on startup or after a control action in the viewer, allclose should be false.
            # when the simulation as settled down, i.e. the targeted site stops moving,
            # allclose will be true.
            if not np.allclose(site_prev, site_world_pos, atol=1e-4, rtol=1e-4):
                settled = False
            else:
                if not settled:
                    # up until now the sim hadn't settled, so evaluate this state.
                    print(f"{site_name}: {site_world_pos}")
                    for n in ["base", "rotation", "pitch", "elbow", "wrist_pitch", "jaw_target"]:
                        n_ = n + "_site"
                        swp = data.site(n_).xpos
                        print(f"{n_:16}: {swp}")

                    # check forward kinetics models
                    qpos = data.qpos.copy()
                    pfw = dh.planar_fk_v3(qpos, False)
                    print(f"planar:           {pfw} @ {qpos[:6]}")
                    with np.printoptions(precision=2):
                        diff = pfw - site_world_pos
                        print(f"planar - target:  {diff}")
                    print(f"qpos rot: {qpos[0]:.4g}, pitch {qpos[1]:.4g}, elbow {qpos[2]:.4g}, wrist_pitch {qpos[3]:.4g}, wrist_roll {qpos[4]:.4g}, jaw {qpos[5]:.4g}")

                    qzero = np.zeros((6,))
                    pfw_zero = dh.planar_fk_v3(qzero, False)
                    print(f"planar zero {pfw_zero} @ {qzero[:4]}")
                settled = True

            site_prev = site_world_pos.copy()
            # body_id = model.body('target_body_name').id
            # current_pos = data.xpos[body_id]

            viewer.sync()

if __name__ == "__main__":
    main()
    
