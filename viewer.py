import argparse
import numpy as np
import mujoco
import mujoco.viewer
import dh
import threed_fk as d3

watchable = ["jaw_grasp", "fixed_jaw_grasp"]
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

    watch = "moving_jaw_grasp" if args.site is None else args.site
    alt = "moving_jaw_grasp"
    with mujoco.viewer.launch_passive(model, data) as viewer:
        watch_site_prev = np.zeros((3,))
        alt_site_prev = np.zeros((3,))
        settled = False  # for convergence
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # monitor watch site, usually moving_jaw target
            watch_site_world_pos = data.site(watch + "_site").xpos
            alt_site_world_pos = data.site(alt + "_site").xpos

            # on startup or after a control action in the viewer, allclose should be false.
            # when the simulation as settled down, i.e. the targeted site stops moving,
            # allclose will be true.
            if not np.allclose(watch_site_prev, watch_site_world_pos, atol=1e-4, rtol=1e-4) or \
               not np.allclose(alt_site_prev, alt_site_world_pos, atol=1e-6, rtol=1e-4):
                settled = False
            else:
                if not settled:
                    print("---------------------------------------------------------------")
                    # up until now the sim hadn't settled, so evaluate this state.
                    sites = ["base", "rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw", "moving_jaw_grasp", "fixed_jaw_grasp", "jaw_grasp"]
                    sites_xpos = {}
                    for n in sites:
                        n_ = n + "_site"
                        swp = data.site(n_).xpos
                        sites_xpos[n] = swp
                        print(f"{n_:22}: {swp[0]:8.5f}, {swp[1]:8.5f}, {swp[2]:8.5f}")

                    # check forward kinetics models
                    qpos = data.qpos.copy()
                    print(f"qpos rot: {qpos[0]:.4g}, pitch {qpos[1]:.4g}, elbow {qpos[2]:.4g}, wrist_pitch {qpos[3]:.4g}, wrist_roll {qpos[4]:.4g}, jaw {qpos[5]:.4g} ({qpos})")
                    all_fk = d3.threed_fk(qpos, False)
                    fkw = all_fk[watch]
                    print(f"planar @{watch}: {fkw} @ qpos {qpos[:6]}")
                    for k, v in all_fk.items():
                        fom = 0
                        emphasis = ""
                        if k in sites_xpos:
                            delta = v - sites_xpos[k]
                            fom = np.linalg.norm(delta, ord=np.inf)
                            if fom > 0.001:
                                emphasis = f"  {delta[0]:8.5f}, {delta[1]:8.5f}, {delta[2]:8.5f}"
                        print(f"{k:16}: {v[0]:8.5f}, {v[1]:8.5f}, {v[2]:8.5f} {fom:0.1e}{emphasis}")

                settled = True

            watch_site_prev = watch_site_world_pos.copy()
            alt_site_prev = alt_site_world_pos.copy()
            # body_id = model.body('target_body_name').id
            # current_pos = data.xpos[body_id]

            viewer.sync()

if __name__ == "__main__":
    main()
    
