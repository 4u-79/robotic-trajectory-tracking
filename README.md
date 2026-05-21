# Dynamic Impedance RL Robot Tracking
This repository contains a trajectory tracking system for a robotic arm. It combines high-level Reinforcement Learning (RL) with low-level Impedance Control to achieve smooth, compliant motion in the presence of noise.

## How to Run the System

### Prerequisites
Make sure you have Python 3.12+ installed. 
### Installation
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/4u-79/robotic-trajectory-tracking.git](https://github.com/4u-79/robotic-trajectory-tracking.git)
   cd robotic-trajectory-tracking
Install the required dependencies:
- pip install -r requirements.txt

Execution
To see the trained agent in action and evaluate its performance against a baseline, run:
 - python evaluate_tracker.py

This script loads the trained model (dynamic_impedance_tracker.zip), runs the simulation, and generates performance metrics.

Tracking Performance:
- The RL agent successfully anticipates curves and adjusts stiffness dynamically, resulting in a much lower tracking error compared to standard rigid control.

Note on Design Choices
The core innovation of this architecture is the separation of control layers into a "Head" and a "Body":
  - The Head which is a High-Level RL Policy: Instead of having the neural network output raw joint torques, the RL agent observes the trajectory error and outputs optimal stiffness and damping parameters.
  - The Body wwhich is a Low-Level Impedance Controller: A classical, physics-based controller takes those dynamically tuned parameters and calculates the actual torques required to move the robot.
This approach solves the "brittleness" problem of standard RL.
By letting classical physics handle the low-level movement, the RL agent is free to act strategically, adjusting the robot's compliance on the fly to absorb noise and smoothly track the target.
