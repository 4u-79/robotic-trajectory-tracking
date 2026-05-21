import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from innovative_tracker import DynamicImpedanceEnv 

def evaluate_and_plot():
    print("Loading Trained Model...")
    env = DynamicImpedanceEnv(render=True)
    
    try:
        model = PPO.load("dynamic_impedance_tracker", device="cpu")
    except FileNotFoundError:
        print("Model not found! Run innovative_tracker.py to train it first.")
        return

    obs, _ = env.reset()
    
    actual_path = []
    target_path = []
    error_log = []

    print("Running evaluation episode...")
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        true_state = env._get_true_state()
        ee_pos = true_state[7:10]
        target_pos = env._get_trajectory(env.t)
        
        actual_path.append(ee_pos)
        target_path.append(target_pos)
        error_log.append(info["error"])
        
        if terminated or truncated:
            break

    env.close()

    print("Generating plots...")
    actual_path = np.array(actual_path)
    target_path = np.array(target_path)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], 
             label="Target Trajectory (Figure-8)", color="blue", linestyle="--")
    ax1.plot(actual_path[:, 0], actual_path[:, 1], actual_path[:, 2], 
             label="Actual Robot Path", color="red", alpha=0.7)
    
    ax1.set_title("3D Cartesian Tracking Performance")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_zlabel("Z (meters)")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(error_log, color="green", linewidth=2)
    
    z = np.polyfit(range(len(error_log)), error_log, 1)
    p = np.poly1d(z)
    ax2.plot(range(len(error_log)), p(range(len(error_log))), 
             "r--", alpha=0.5, label="Error Trend")

    ax2.set_title("Tracking Error Over Time (Under Noise & Delay)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Euclidean Distance Error (meters)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("tracking_results.png", dpi=300)
    print("Saved plot as 'tracking_results.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()