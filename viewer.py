import argparse
import numpy as np
import mujoco
import mujoco.viewer
import dh
import threed_fk as d3

watchable = ["jaw_target", "fixed_jaw"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-g", action="store_true", help="simulate with no gravity")
    parser.add_argument("--site", choices=watchable, help="site")
    args = parser.parse_args()
    print(f"--zero-g {args.zero_g}")
    model = mujoco.MjModel.from_xml_path('/home/greg/gym-so100-c/gym_so100/assets/so100_transfer_cube.xml')
    data = mujoco.MjData(model)
    if args.zero_g:
        model.opt.gravity[:] = 0

    watch_site = "jaw_target" if args.site is None else args.site

    with mujoco.viewer.launch_passive(model, data) as viewer:
        site_prev = np.zeros((3,))
        fj_site_prev = np.zeros((3,))
        settled = False  # for convergence
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # monitor jaw target
            site_world_pos = data.site(watch_site + "_site").xpos
            fj_site_world_pos = data.site("fixed_jaw" + "_site").xpos

            # on startup or after a control action in the viewer, allclose should be false.
            # when the simulation as settled down, i.e. the targeted site stops moving,
            # allclose will be true.
            if not np.allclose(site_prev, site_world_pos, atol=1e-4, rtol=1e-4) or \
               not np.allclose(fj_site_prev, fj_site_world_pos, atol=1e-6, rtol=1e-4):
                settled = False
            else:
                if not settled:
                    # up until now the sim hadn't settled, so evaluate this state.
                    print(f"{watch_site}: {site_world_pos}")
                    for n in ["base", "rotation", "pitch", "elbow", "wrist_pitch", "fixed_jaw", "jaw_target"]:
                        n_ = n + "_site"
                        swp = data.site(n_).xpos
                        print(f"{n_:16}: {swp}")

                    # check forward kinetics models
                    qpos = data.qpos.copy()
                    all = d3.threed_fk(qpos, False)
                    pfw = all['jaw_target']
                    print(f"planar:           {pfw} @ {qpos[:6]}")
                    with np.printoptions(precision=2):
                        diff = pfw - site_world_pos
                        print(f"planar - {watch_site}: {np.linalg.norm(diff):.3g} {diff}")
                    print(f"qpos rot: {qpos[0]:.4g}, pitch {qpos[1]:.4g}, elbow {qpos[2]:.4g}, wrist_pitch {qpos[3]:.4g}, wrist_roll {qpos[4]:.4g}, jaw {qpos[5]:.4g}")
                    for k, v in all.items():
                        print(f"{k:16}: {v}")

                    qzero = np.zeros((6,))
                    pfw_zero = d3.threed_fk(qzero, False)['jaw_target']
                    print(f"planar zero {pfw_zero} @ {qzero[:4]}")
                settled = True

            site_prev = site_world_pos.copy()
            fj_site_prev = fj_site_world_pos.copy()
            # body_id = model.body('target_body_name').id
            # current_pos = data.xpos[body_id]

            viewer.sync()

if __name__ == "__main__":
    main()
    
