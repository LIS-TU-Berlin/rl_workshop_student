from typing import Optional

import mujoco
import numpy as np
import time, math
import robotic as ry
import mujoco_warp as mjw
import warp as wp

class MujocoSim:
    """
    Wraps the Mujoco simulator.
    
    Assumes that the simulation is stepped with a high frequency, while the
    policy provides new controls with a lower frequency. The high frequency stepping is
    read out from a control buffer, which are explicit and continuously/smoothly varying motor
    references for each micro step. This control buffer needs to be re-filled before each
    multi-step. The policy output typically defines a spline (see poly_ref.py)
    which is sampled with the high frequency to fill that control buffer.

    Further, this wrapper handles qoffsets: When the free objects in the original configuration
    have parent (i.e. non-zero origin) frames, this implies a qpos offset. That's the case
    esp. when the configuration is a multi-scene configuration.
    """
    mj_steps: int = 0
    ctrl_time: float = 0.
    ctrl_costs: float = 0.

    # for inspection & rendering only
    view_speed: float = -1.
    view_text: str = ''
    saved_qpos: Optional[list] = None
    saved_images: Optional[list] = None #initialize to [] to activate recording

    def __init__(
        self,
        xml_description: str,
        C: ry.Config,
        tau_sim: float = 1e-3,
        warp_worlds: int = 0,
        use_mj_viewer: bool = False,
    ):
        """
        Basic simulation class that wraps a mujoco simulator.
        """
        self.model = mujoco.MjModel.from_xml_string(xml_description)
        if warp_worlds>0:
            self.wp_model = mjw.put_model(self.model)
            self.data = mjw.make_data(self.model, nworld=warp_worlds)
        else:
            self.model = mujoco.MjModel.from_xml_string(xml_description)
            self.data = mujoco.MjData(self.model)
        self.warp_worlds = warp_worlds
        self.tau_sim = tau_sim
        self.model.opt.timestep = self.tau_sim
        self.use_mj_viewer = use_mj_viewer

        # mostly not used, but still useful to double check and debug ry->mj conversion
        if use_mj_viewer:
            C.view(False)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            cam_pose = C.get_viewer().getCamera_pose()
            q = ry.Quaternion()
            q.set(cam_pose[3:])
            rpy = q.getRollPitchYaw()
            self.viewer.cam.azimuth = -90.+rpy[1]*180./math.pi
            self.viewer.cam.elevation = 90.-rpy[0]*180./math.pi
            self.viewer.cam.distance = 2.
            self.viewer.cam.lookat = [0., 0., .5]
        else:
            self.viewer = None

        self.C = C

        # determine which qpos indices are controlled (actuated) joints
        self.ctrl_indices = []
        for i in range(self.model.nu):
            id = self.model.actuator(i).trnid[0]
            qid = self.model.joint(id).qposadr
            self.ctrl_indices.append(int(qid[0]))
        self.ctrl_indices = np.array(self.ctrl_indices, dtype='int32')

        self.ctrl_buffer = None
        self.ctrl_bufferPtr = 0
        self.ctrlRef_spline = None

        # determine free objects and their non-zero offset in qpos
        self.qpos_offset = np.zeros((self.model.nq))
        self.freeobjs = []
        for f in self.C.getFrames():
            parent = f.getParent()
            if parent == None or (f.getJointType() == ry.JT.free):
                if "mass" in f.asDict():
                    self.freeobjs.append(f)
                    if parent is not None:
                        assert parent.getParent() is None, "free joints need to have a root parent!"
                        offset = parent.getPosition()
                        quat = parent.getQuaternion()
                        assert np.linalg.norm(quat - np.array([1,0,0,0]))<1e-10
                        qid = f.getJointQIndex()
                        self.qpos_offset[qid:qid+3] = offset
        if self.warp_worlds>0:
            self.qpos_offset = np.tile(self.qpos_offset, (self.warp_worlds, 1))

        print(f"-- initialized MjSim with (controlled) joint dimension {C.getJointDimension()} (mj qpos:{self.data.qpos.size} qvel:{self.data.qvel.size} ctrl:{self.model.nu}) with {len(self.freeobjs)} free objects") # ctrl_indices:{self.ctrl_indices}
        
        if self.warp_worlds>0:
            assert self.data.qpos.size == self.C.getJointDimension() * self.warp_worlds
        else:
            assert self.data.qpos.size == self.C.getJointDimension()

        if self.warp_worlds>0:
            self.set_state(np.tile(self.C.getJointState(), (self.warp_worlds, 1)))
        else:
            self.set_state(self.C.getJointState())

    def __del__(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def multi_step(self, num_steps: int) -> None:
        view_steps = math.ceil(0.02 / self.tau_sim * self.view_speed)
        if self.warp_worlds>0 and isinstance(self.ctrl_buffer, np.ndarray):
            wp_ctrl_buffer = wp.array(self.ctrl_buffer, dtype=wp.float32)

        for k in range(num_steps):
            assert self.ctrl_buffer is not None
            assert self.ctrl_buffer.shape[0]>self.ctrl_bufferPtr , 'ctrlRef buffer too small'

            if self.warp_worlds>0:
                self.data.ctrl = wp_ctrl_buffer[self.ctrl_bufferPtr]
            else:
                self.data.ctrl = self.ctrl_buffer[self.ctrl_bufferPtr].reshape(-1)
            self.ctrl_bufferPtr += 1

            if self.warp_worlds>0:
                mjw.step(self.wp_model, self.data)
            else:
                mujoco.mj_step(self.model, self.data)
            self.mj_steps += 1
            self.ctrl_time += self.tau_sim
            if self.warp_worlds>0:
                pass
            else:
                self.ctrl_costs += np.sum(np.square(self.data.actuator_force))

            # storing the path
            if self.saved_qpos is not None:
                self.saved_qpos.append(self.data.qpos.copy())

            # visualization, and storing images
            if self.view_speed > 0.0 and (self.mj_steps%view_steps==0):
                if self.use_mj_viewer:
                    self.viewer.sync()
                if self.warp_worlds>0:
                    self.C.setJointState(self._qpos[0])
                else:
                    self.C.setJointState(self._qpos)
                self.C.view(False, f"{self.view_text} simulation t: {self.ctrl_time:6.3f}", offscreen=(self.saved_images is not None))
                if self.saved_images is not None:
                    self.saved_images.append(self.C.get_viewer().getRgb())
                else:
                    time.sleep(view_steps * self.tau_sim / self.view_speed)

        if self.warp_worlds>0:
            mjw.forward(self.wp_model, self.data)
            self.C.setJointState(self._qpos[0])
        else:
            mujoco.mj_forward(self.model, self.data)
            self.C.setJointState(self._qpos)

    def step(self, tau_step: float) -> None:
        """[core] step the physics engine for a given time, usually making multiple small (tau_sim) steps"""
        sim_steps = round(tau_step / self.tau_sim)
        assert math.isclose(tau_step, sim_steps * self.tau_sim), "tau_step needs to be a multiple of tau_sim"
        self.multi_step(sim_steps)

    def set_state(
        self,
        qpos: np.ndarray,
        qvel: Optional[np.ndarray] = None,
        act: Optional[np.ndarray] = None,
        world_id: int = -1,
    ) -> None:
        if self.warp_worlds==0:
            self.data.qpos = qpos.reshape(-1) + self.qpos_offset
            self.C.setJointState(qpos)
            if qvel is None:
                self.data.qvel *= 0.
            else:
                self.data.qvel = qvel.reshape(-1)
            if act is None:
                self.data.actuator_force *= 0.
            else:
                self.data.actuator_force = act.reshape(-1)
            mujoco.mj_forward(self.model, self.data)
        else:
            if world_id==-1:
                qpos = qpos.reshape(self.warp_worlds, -1)
                self.data.qpos = wp.array(qpos + self.qpos_offset, dtype=wp.float32)
                self.C.setJointState(qpos[0])
            else:
                self.data.qpos[world_id] = wp.array(qpos + self.qpos_offset, dtype=wp.float32)
                self.C.setJointState(qpos)

            if qvel is None:
                self.data.qvel *= 0.
            else:
                self.data.qvel = wp.array(qvel.reshape(self.warp_worlds, -1), dtype=wp.float32)
            if act is None:
                self.data.actuator_force *= 0.
            else:
                self.data.actuator_force = wp.array(act.reshape(self.warp_worlds, -1), dtype=wp.float32)
            mjw.forward(self.wp_model, self.data)

    def get_Jacobian(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # body_id = mujoco.mj_name2id(self.model, 1, frame_name)
        # assert body_id>=0, f'frame name {frame_name} is not a mj body'

        Jpos = np.empty((len(body_ids), 3, self.model.nv))
        Jang = np.empty((len(body_ids), 3, self.model.nv))

        for i, body_id in enumerate(body_ids.reshape(-1)):
            pos = self.data.xpos[body_id]
            mujoco.mj_jac(self.model, self.data, Jpos[i], Jang[i], pos, body_id)
    
        if False: #test
            self.C.setJointState(self._qpos)
            y, J = self.C.eval(ry.FS.position, [frame_name])
            print(np.linalg.norm(pos-y))
            print(Jpos, '\n', J)

        return Jpos, Jang

    def get_pts(self, pt_ids: np.ndarray, world_id: int = 0) -> np.ndarray:
        if self.warp_worlds>0:
            raise Exception('NIY')
        else:
            return self.data.xpos[pt_ids]

    def get_pt_ids(self, pt_names: list[str]) -> list[int]:
        ids = []
        for name in pt_names:
            for i in range(self.model.nbody):
                b_name = self.model.body(i).name
                if b_name.endswith(name):
                    ids.append(i)
        return ids
       
    @property
    def _qpos(self) -> np.ndarray:
        if self.warp_worlds>0:
            return self.data.qpos.numpy() - self.qpos_offset
        else:
            return self.data.qpos - self.qpos_offset
