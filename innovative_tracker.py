import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from collections import deque
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
class DynamicImpedanceEnv(gym.Env):
    """
    An advanced RL environment where the agent tunes the robot's physical compliance 
    (stiffness/damping) and virtual target offsets, rather than raw motor velocities.
    """
    def __init__(self, render=False):
        super(DynamicImpedanceEnv, self).__init__()
        
        # ACTION SPACE (5 Dimensions)
        # [0, 1, 2] -> Virtual Cartesian Offset (agent slightly shifts the target to anticipate lag)
        # [3] -> Dynamic Stiffness (Kp) multiplier
        # [4] -> Dynamic Damping (Kd) multiplier
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # STATE SPACE (16 Dimensions)
        # Current Joint Pos (7), Current EE Pos (3), Target EE Pos (3), Target Velocity Vector (3)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(16,), dtype=np.float32)
        
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Uncertainty requirement: 3-step Observation Delay Buffer
        self.obs_buffer = deque(maxlen=3) 
        self.robot = None
        self.t = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Load KUKA iiwa (7-DOF)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.t = 0
        
        # Fill delay buffer
        obs = self._get_true_state()
        for _ in range(3):
            self.obs_buffer.append(obs)
            
        return self.obs_buffer[0], {} # Return delayed observation
        
    def _get_trajectory(self, t):
        # A complex 3D figure-eight (Lissajous curve) to test advanced tracking
        speed = 0.05
        x = 0.5 + 0.2 * np.sin(t * speed)
        y = 0.2 * np.sin(2 * t * speed)
        z = 0.4 + 0.1 * np.cos(t * speed)
        return np.array([x, y, z])
        
    def _get_true_state(self):
        # Joints
        joint_states = p.getJointStates(self.robot, range(7))
        joint_positions = np.array([s[0] for s in joint_states])
        
        # EE Position with injected noise (Uncertainty requirement)
        ee_state = p.getLinkState(self.robot, 6)
        ee_pos = np.array(ee_state[0]) + np.random.normal(0, 0.002, 3) 
        
        # Trajectory Geometry (Position & Velocity Vector)
        curr_target = self._get_trajectory(self.t)
        next_target = self._get_trajectory(self.t + 1)
        target_vel = next_target - curr_target # Gives the agent the tangent of the curve
        
        return np.concatenate([joint_positions, ee_pos, curr_target, target_vel]).astype(np.float32)

    def step(self, action):
        # 1. Decode Action Space
        target_offset = action[0:3] * 0.05 # Max 5cm offset
        stiffness_action = np.interp(action[3], [-1, 1], [0.01, 0.1]) # Map to PyBullet Kp limits
        damping_action = np.interp(action[4], [-1, 1], [0.5, 1.0])    # Map to PyBullet Kd limits
        
        # 2. Virtual Target Computation
        true_target = self._get_trajectory(self.t)
        virtual_target = true_target + target_offset
        
        # 3. Inverse Kinematics based on Virtual Target
        target_joint_angles = p.calculateInverseKinematics(
            self.robot, 6, virtual_target, maxNumIterations=100
        )
        
        # 4. Apply dynamically tuned PID control
        # The agent changes the physical compliance of the robot on the fly
        p.setJointMotorControlArray(
            self.robot, range(7), p.POSITION_CONTROL, 
            targetPositions=target_joint_angles,
            positionGains=[stiffness_action]*7,
            velocityGains=[damping_action]*7
        )
        
        p.stepSimulation()
        self.t += 1
        
        # 5. Get State & Manage Delay Queue
        true_state = self._get_true_state()
        self.obs_buffer.append(true_state)
        delayed_obs = self.obs_buffer[0] # Agent only sees 3 timesteps ago
        
        # --- REWARD CALCULATION ---
        ee_pos = true_state[7:10]
        distance = np.linalg.norm(ee_pos - true_target)
        
        # Primary objective: Minimize distance
        reward = -distance 
        
        # Elegance objective: Penalize high stiffness. 
        # Forces agent to use the minimum required muscle tension, absorbing noise naturally.
        reward -= 0.01 * stiffness_action 
        
        terminated = False
        truncated = self.t >= 500
        
        return delayed_obs, reward, terminated, truncated, {"error": distance}

if __name__ == "__main__":
    print("Initializing Asymmetric Dynamic Tracking with Multi-Processing...")
    
    # 1. Optimize PyTorch math operations for your Ryzen architecture
    torch.set_num_threads(8) 
    
    # 2. Define how many CPU cores to use for parallel physics simulation
    num_cpu = 8  # Ryzen 7 has 8 physical cores
    
    # 3. Create a factory function to spawn multiple environments
    def make_env():
        def _init():
            return DynamicImpedanceEnv(render=False)
        return _init
        
    # 4. Spin up 8 parallel PyBullet worlds across your CPU threads
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])
    
    # 5. Initialize PPO (We scale the batch size to handle the 8x data influx)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        device="cpu",
        batch_size=64 * num_cpu, 
        n_steps=2048
    )
    
    print(f"Training started across {num_cpu} CPU cores...")
    model.learn(total_timesteps=1_500_000, progress_bar=True)
    model.save("dynamic_impedance_tracker")
    print("Model saved. Ready for evaluation.")