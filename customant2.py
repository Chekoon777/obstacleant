import math,sys
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym import utils
from gymnasium.spaces import Box

def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z

class CustomAntEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="/home/chek/workspace/customant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs):

        self.headingCoef = 1e-4 # Heading penalty coef for reward

        print ("Custom Ant Environment made by SJ.")
        if sys.platform == 'darwin': # OSX
            xmlPath = '/Users/sungjoon/github/mujoco-ant/src/ant_custom.xml'
        elif sys.platform == 'linux':
            xmlPath = '/home/chek/workspace/customant_stairs.xml'
        else:
            print ("Unknown platform: [%s]."%(xmlPath))
        
        obs_shape = 27
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        # xmlPath = os.getcwd()+'/ant_custom.xml'
        MujocoEnv.__init__(self, xmlPath, observation_space=observation_space, frame_skip=5)
        utils.EzPickle.__init__(self)
    

        # Do reset once 
        self.reset()

        # Some parameters
        # self.minPosDeg = np.array([-30,30,-30,-70,-30,-70,-30,30])
        # self.maxPosDeg = np.array([+30,70,+30,-30,+30,-30,+30,70])

        D1 = 30
        D2 = 70
        DX = 80
        self.minPosDeg = np.array([-DX,D1,-DX,-D2,-DX,-D2,-DX,D1])
        self.maxPosDeg = np.array([+DX,D2,+DX,-D1,+DX,-D1,+DX,D2])

        # Observation and action dimensions 
        self.obsDim = self.observation_space.shape[0]
        self.actDim = self.action_space.shape[0]

    def step(self, a):
        headingBefore = self.get_heading()
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip) # Run!
        headingAfter = self.get_heading()
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        forward_reward = (xposafter - xposbefore)/self.dt 
        # Modified (upperbound on forward reward)
        if forward_reward > 1.0: 
            forward_reward = 1.0
        # Heading cost
        heading_cost = self.headingCoef*(headingAfter**2+yposafter**2)
        # Control cost
        ctrl_cost = .5 * np.square(a).sum()
        # Contact cost
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # Survive 
        survive_reward = 1.0 
        reward = forward_reward - heading_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done,\
            dict(
                reward_forward=forward_reward,
                reward_heading=-heading_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + 0*self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + 0*self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 3.0

    def get_heading(self):
        q = self.data.get_body_xquat('torso')
        rX,rY,rZ = quaternion_to_euler_angle(q[0],q[1],q[2],q[3])
        return rZ