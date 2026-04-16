import os
import sys
import copy
import numpy as np
import yaml
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class WheelfootController:
    def __init__(self, model_dir, robot, robot_type, rl_type, start_controller):
        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.rl_type = rl_type
        self.is_mjlab = self.rl_type == "mjlab"

        # Load configuration and model file paths based on robot type
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model_policy = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/policy.onnx'
        self.model_encoder = None if self.is_mjlab else f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/encoder.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)
        
        # Load the ONNX model
        self.initialize_onnx_models()

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 1
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller

        # Gait index
        self.gait_index = 0

        # Flag indicating first received observation
        self.is_first_rec_obs = True

        # Observation
        self.fake_pose_cmd = np.array([0.0, 0.0, 1.0, 0.0, 1.0])

    def initialize_onnx_models(self):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']
        
        # Load the ONNX model and set up input and output names
        self.policy_session = ort.InferenceSession(self.model_policy, sess_options=session_options, providers=cpu_providers)
        self.policy_input_names = [self.policy_session.get_inputs()[i].name for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_names = [self.policy_session.get_outputs()[i].name for i in range(self.policy_session.get_outputs().__len__())]
        self.policy_input_shapes = [self.policy_session.get_inputs()[i].shape for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_shapes = [self.policy_session.get_outputs()[i].shape for i in range(self.policy_session.get_outputs().__len__())]

        if self.is_mjlab:
            self.encoder_session = None
            self.encoder_input_names = []
            self.encoder_output_names = []
            self.encoder_input_shapes = []
            self.encoder_output_shapes = []
            self.validate_mjlab_policy()
        else:
            self.encoder_session = ort.InferenceSession(self.model_encoder, sess_options=session_options, providers=cpu_providers)
            self.encoder_input_names = [self.encoder_session.get_inputs()[i].name for i in range(self.encoder_session.get_inputs().__len__())]
            self.encoder_output_names = [self.encoder_session.get_outputs()[i].name for i in range(self.encoder_session.get_outputs().__len__())]
            self.encoder_input_shapes = [self.encoder_session.get_inputs()[i].shape for i in range(self.encoder_session.get_inputs().__len__())]
            self.encoder_output_shapes = [self.encoder_session.get_outputs()[i].shape for i in range(self.encoder_session.get_outputs().__len__())]

    def validate_mjlab_policy(self):
        expected_shapes = {
            'obs_history': [1, self.obs_history_length * self.observations_size],
            'obs': [1, self.observations_size],
            'commands': [1, self.mjlab_policy_commands_size],
        }
        if len(self.policy_input_names) != 3:
            raise ValueError(f"mjlab policy expects 3 inputs, got {self.policy_input_names}")

        for input_name, input_shape in zip(self.policy_input_names, self.policy_input_shapes):
            if input_name not in expected_shapes:
                raise ValueError(f"Unexpected mjlab policy input '{input_name}'")
            if list(input_shape) != expected_shapes[input_name]:
                raise ValueError(
                    f"mjlab policy input '{input_name}' shape {input_shape} does not match expected {expected_shapes[input_name]}"
                )

        if len(self.policy_output_shapes) != 1 or list(self.policy_output_shapes[0]) != [1, self.actions_size]:
            raise ValueError(
                f"mjlab policy output shape {self.policy_output_shapes} does not match expected [[1, {self.actions_size}]]"
            )
        print(
            f"[mjlab] policy io checked: obs={self.observations_size}, "
            f"obs_history={self.obs_history_length * self.observations_size}, "
            f"actions={self.actions_size}, commands={self.mjlab_policy_commands_size}"
        )

    # Load the configuration from a YAML file
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        pointfoot_cfg = config['PointfootCfg']
        mjlab_cfg = pointfoot_cfg.get('mjlab', {})
        size_cfg = copy.deepcopy(pointfoot_cfg['size'])
        self.control_cfg = copy.deepcopy(pointfoot_cfg['control'])
        if self.is_mjlab:
            self.control_cfg.update(mjlab_cfg.get('control', {}))
            size_cfg.update(mjlab_cfg.get('size', {}))

        # Assign configuration parameters to controller variables
        self.joint_names = pointfoot_cfg['joint_names']
        self.init_state = pointfoot_cfg['init_state']['default_joint_angle']
        self.stand_duration = pointfoot_cfg['stand_mode']['stand_duration']
        self.rl_cfg = pointfoot_cfg['normalization']
        self.obs_scales = self.rl_cfg['obs_scales']
        self.actions_size = size_cfg['actions_size']
        self.commands_size = size_cfg['commands_size']
        self.observations_size = size_cfg['observations_size']
        self.obs_history_length = size_cfg['obs_history_length']
        self.encoder_output_size = size_cfg['encoder_output_size']
        self.imu_orientation_offset = np.array(list(pointfoot_cfg['imu_orientation_offset'].values()))
        self.user_cmd_cfg = pointfoot_cfg['user_cmd_scales']
        self.user_cmd_offsets = pointfoot_cfg.get('user_cmd_offsets', {
            'lin_vel_x': 0.0,
            'lin_vel_y': 0.0,
            'ang_vel_yaw': 0.0,
        })
        self.loop_frequency = pointfoot_cfg['loop_frequency']
        self.encoder_input_size = self.obs_history_length * self.observations_size

        # Initialize variables for actions, observations, and commands
        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.command_actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(self.commands_size)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.base_pose_commands = np.zeros(4)
        self.base_se3_decrease_rate = np.zeros(1)
        self.mjlab_history_buffers = []
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints

        self.joint_pos_idxs = size_cfg['jointpos_idxs']
        self.wheel_joint_damping = self.control_cfg['wheel_joint_damping']
        self.wheel_joint_torque_limit = self.control_cfg['wheel_joint_torque_limit']
        self.action_scale_vel = self.control_cfg.get('action_scale_vel', 5.0)

        mjlab_command_cfg = mjlab_cfg.get('command', {})
        self.mjlab_base_pose_scale = mjlab_command_cfg.get('base_pose_scale_xy', 2.0)
        self.mjlab_base_se3_decrease_rate = mjlab_command_cfg.get('base_se3_decrease_rate', 1.0)
        self.mjlab_target_orient_x_axis = np.array(
            mjlab_command_cfg.get('target_orient_x_axis', [1.0, 0.0]),
            dtype=np.float32,
        )
        self.mjlab_command_offsets = np.array([
            mjlab_command_cfg.get('input_offsets', {}).get('lin_vel_x', self.user_cmd_offsets['lin_vel_x']),
            mjlab_command_cfg.get('input_offsets', {}).get('lin_vel_y', self.user_cmd_offsets['lin_vel_y']),
            mjlab_command_cfg.get('input_offsets', {}).get('ang_vel_yaw', self.user_cmd_offsets['ang_vel_yaw']),
        ])
        self.mjlab_cmd_offsets = np.array([
            mjlab_command_cfg.get('cmd_offsets', {}).get('lin_vel_x', 0.0),
            mjlab_command_cfg.get('cmd_offsets', {}).get('lin_vel_y', 0.0),
            mjlab_command_cfg.get('cmd_offsets', {}).get('ang_vel_yaw', 0.0),
        ])
        self.mjlab_command_deadband = mjlab_command_cfg.get('input_deadband', 0.05)
        self.mjlab_policy_commands_size = mjlab_command_cfg.get('policy_commands_size', 0)
        self.mjlab_history_term_dims = mjlab_command_cfg.get('history_term_dims', [3, 3, 3, 6, 8, 8])
        if self.is_mjlab and sum(self.mjlab_history_term_dims) != self.observations_size:
            raise ValueError(
                f"mjlab history term dims {self.mjlab_history_term_dims} do not sum to observations_size {self.observations_size}"
            )
        self.policy_commands = np.zeros(self.mjlab_policy_commands_size)

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]
        
        # Set initial mode to "STAND"
        self.mode = "STAND"
    
    # Main control loop
    def run(self):
        # Wait until the controller is started
        while not self.start_controller:
          time.sleep(1)

        # Initialize default joint angles for standing
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        # Set the loop rate based on the frequency in the configuration
        rate = Rate(self.loop_frequency)
        while self.start_controller:
            self.update()
            rate.sleep()
        
        # Reset robot command values to ensure a safe stop when exiting the loop
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [1.0 for x in range(0, self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    # Handle the stand mode for smoothly transitioning the robot into standing
    def handle_stand_mode(self):
        self.init_state["hip_L_Joint"] = -0.9
        self.init_state["hip_R_Joint"] = 0.9
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                if (j + 1) % 4 != 0:
                    # Interpolate between initial and default joint angles during stand mode
                    pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[self.joint_names[j]] * self.stand_percent
                    self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
                else:
                    self.set_joint_command(0, 0, 0, self.wheel_joint_damping, 0, 0)
            # Increment the stand percentage over time
            self.stand_percent += 3 / (self.stand_duration * self.loop_frequency)
        else:
            # Switch to walk mode after standing
            self.mode = "WALK"

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        self.init_joint_angles[1] = 0.0
        self.init_joint_angles[5] = 0.0

        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_encoder()
            self.compute_actions()
            # Clip the actions within predefined limits
            if not self.is_mjlab:
                action_min = -self.rl_cfg['clip_scales']['clip_actions']
                action_max = self.rl_cfg['clip_scales']['clip_actions']
                self.actions = np.clip(self.actions, action_min, action_max)

            if self.is_mjlab:
                self.last_actions = np.array(self.actions)
                self.command_actions = self.reorder_mjlab_actions_to_robot(self.actions)
            elif self.rl_type == "isaaclab":
                self.command_actions = self.swap_positions(self.actions, reverse=True)
            else:
                self.command_actions = np.array(self.actions)

        command_actions = np.array(self.command_actions)

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:
                # Compute the limits for the action based on joint position and velocity
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # Clip action within limits
                action_value = max(action_min / self.control_cfg['action_scale_pos'],
                                   min(action_max / self.control_cfg['action_scale_pos'], command_actions[i]))

                # Compute the desired joint position and set it
                pos_des = action_value * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

                # Save the last action for reference
                if not self.is_mjlab:
                    self.last_actions[i] = action_value
            else:
                velocity_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                velocity_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping

                if self.is_mjlab:
                    velocity_des = np.clip(command_actions[i] * self.action_scale_vel, velocity_min, velocity_max)
                else:
                    self.last_actions[i] = command_actions[i]
                    action_value = max(velocity_min / self.wheel_joint_damping,
                                       min(velocity_max / self.wheel_joint_damping, command_actions[i]))
                    velocity_des = action_value * self.action_scale_vel * self.wheel_joint_damping

                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def swap_positions(self, initial_array, reverse=False, exclude_wheel=False):
        if not exclude_wheel:
            joint_idx_lab = [0, 4, 1, 5, 2, 6, 3, 7]
        else:
            joint_idx_lab = [0, 3, 1, 4, 2, 5]
        new_array = np.zeros(initial_array.shape)
        for i in range(len(joint_idx_lab)):
            if not reverse:
                new_array[i] = initial_array[joint_idx_lab[i]]
            else:
                new_array[joint_idx_lab[i]] = initial_array[i]
        return new_array

    def reorder_mjlab_actions_to_robot(self, policy_actions):
        robot_actions = np.zeros(policy_actions.shape, dtype=policy_actions.dtype)
        robot_actions[0:3] = policy_actions[0:3]
        robot_actions[3] = policy_actions[6]
        robot_actions[4:7] = policy_actions[3:6]
        robot_actions[7] = policy_actions[7]
        return robot_actions

    def apply_deadband(self, value, deadband):
        if abs(value) < deadband:
            return 0.0
        return value

    def update_mjlab_history(self, obs_terms):
        if self.is_first_rec_obs:
            self.mjlab_history_buffers = [
                np.tile(term.reshape(1, -1), (self.obs_history_length, 1)) for term in obs_terms
            ]
            self.is_first_rec_obs = False
        else:
            for buffer, term in zip(self.mjlab_history_buffers, obs_terms):
                buffer[:-1] = buffer[1:]
                buffer[-1] = term

        self.proprio_history_vector = np.concatenate([buffer.reshape(-1) for buffer in self.mjlab_history_buffers])
        self.proprio_history_buffer = np.array(self.proprio_history_vector)
    
    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # Quaternion to Euler ZYX conversion
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # Get the inverse rotation matrix

        # Project the gravity vector (pointing downwards) into the body frame
        gravity_vector = np.array([0, 0, -1])  # Gravity in world frame (z-axis down)
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # Transform gravity into body frame

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # Apply IMU orientation offset correction (using Euler angles)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # Rotation matrix for offset correction
        base_ang_vel = np.dot(rot, base_ang_vel)  # Apply correction to angular velocity
        projected_gravity = np.dot(rot, projected_gravity)  # Apply correction to projected gravity

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # Retrieve the last actions that were applied to the robot
        actions = np.array(self.last_actions)

        # Create a command scaler matrix for linear and angular velocities
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # Scale factor for linear velocity in x direction
            self.user_cmd_cfg['lin_vel_y'],  # Scale factor for linear velocity in y direction
            self.user_cmd_cfg['ang_vel_yaw']  # Scale factor for yaw (angular velocity)
        ])
        command_offsets = np.array([
            self.user_cmd_offsets['lin_vel_x'],
            self.user_cmd_offsets['lin_vel_y'],
            self.user_cmd_offsets['ang_vel_yaw'],
        ])

        # Apply scaling to the command inputs (velocity commands)
        self.scaled_commands = np.dot(command_scaler, self.commands) + command_offsets

        # Populate observation vector
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # In WF, joint pos does not include wheel speed, index(3, 7) needs to be removed
        joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])

        if self.is_mjlab:
            effective_commands = np.clip(self.commands + self.mjlab_cmd_offsets, -1.0, 1.0)
            self.scaled_commands = np.array(effective_commands)

            # mjlab velocity-tracking actor obs layout (31):
            # [velocity_commands(3), base_ang_vel(3), proj_gravity(3), joint_pos(6), joint_vel(8), last_action(8)]
            obs_terms = [
                self.scaled_commands,
                base_ang_vel * self.obs_scales['ang_vel'],
                projected_gravity,
                joint_pos_input,
                joint_velocities * self.obs_scales['dof_vel'],
                actions,
            ]
            obs = np.concatenate(obs_terms)
            self.update_mjlab_history(obs_terms)
            self.observations = obs
            return

        # swap positions in joint_pos, joint_vel and actions if mode is isaaclab
        if self.rl_type == "isaaclab":
            joint_pos_input = self.swap_positions(joint_pos_input, exclude_wheel=True)
            joint_velocities = self.swap_positions(joint_velocities)
            actions = self.swap_positions(actions)

        # Create the observation vector by concatenating various state variables:
        # - Base angular velocity (scaled)
        # - Projected gravity vector
        # - Joint positions (difference from initial angles, scaled)
        # - Joint velocities (scaled)
        # - Last actions applied to the robot
        # - Scaled command inputs
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],  # Scaled base angular velocity
            projected_gravity,  # Projected gravity vector in body frame
            joint_pos_input,  # Scaled joint positions
            joint_velocities * self.obs_scales['dof_vel'],  # Scaled joint velocities
            actions  # Last actions taken by the robot
        ])

        # Check if this is the first recorded observation
        if self.is_first_rec_obs:
            # Calculate the total size of the encoder input, handling dynamic batch dimensions
            shape = self.encoder_input_shapes[0]
            # Filter out dynamic dimensions (strings) and replace with batch size of 1
            numeric_shape = [1 if isinstance(dim, str) else dim for dim in shape]
            input_size = np.prod(numeric_shape)
            
            # Initialize the proprioceptive history buffer with zeros
            self.proprio_history_buffer = np.zeros(input_size)

            # Fill the proprioceptive history buffer with the current observation for the entire history length
            for i in range(self.obs_history_length):
                self.proprio_history_buffer[i * self.observations_size:(i + 1) * self.observations_size] = obs

            # Update the flag to indicate that the first observation has been processed
            self.is_first_rec_obs = False
        
        # Shift the existing proprioceptive history buffer to the left
        self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size:]

        # Add the current observation to the end of the proprioceptive history buffer
        self.proprio_history_buffer[-self.observations_size:] = obs

        # Convert the proprioceptive history buffer to a numpy array
        self.proprio_history_vector = np.array(self.proprio_history_buffer)

        # Clip the observation values to within the specified limits for stability
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # Lower limit for clipping
            self.rl_cfg['clip_scales']['clip_observations']  # Upper limit for clipping
        )

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        if self.is_mjlab:
            inputs = {
                'obs_history': self.proprio_history_vector.astype(np.float32).reshape(1, -1),
                'obs': self.observations.astype(np.float32).reshape(1, -1),
                'commands': self.policy_commands.astype(np.float32).reshape(1, -1),
            }
            output = self.policy_session.run(self.policy_output_names, inputs)
            self.actions = np.array(output[0]).flatten()
            return

        # Concatenate observations into a single tensor and convert to float32
        input_tensor = np.concatenate([self.encoder_out, self.observations, self.fake_pose_cmd, self.scaled_commands], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        # Add batch dimension for ONNX model (reshape from [features] to [1, features])
        input_tensor = input_tensor.reshape(1, -1)

        # Create a dictionary of inputs for the policy session
        inputs = {self.policy_input_names[0]: input_tensor}

        # Run the policy session and get the output
        output = self.policy_session.run(self.policy_output_names, inputs)

        # Flatten the output and store it as actions
        self.actions = np.array(output).flatten()

    def compute_encoder(self):
        """
        Computes the encoder output based on the proprioceptive history buffer.

        This method first concatenates the proprioceptive history buffer into a single input tensor.
        Then it converts the input tensor to the float32 data type. After that, it creates a dictionary
        of inputs for the encoder session and runs the encoder session to get the output. Finally,
        it flattens the output and stores it as the encoder output.
        """
        if self.is_mjlab:
            return

        # Concatenate the proprioceptive history buffer into a single tensor and convert to float32
        input_tensor = np.concatenate([self.proprio_history_buffer], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        # Add batch dimension for ONNX model (reshape from [features] to [1, features])
        input_tensor = input_tensor.reshape(1, -1)

        # Create a dictionary of inputs for the encoder session
        inputs = {self.encoder_input_names[0]: input_tensor}

        # Run the encoder session and get the output
        output = self.encoder_session.run(self.encoder_output_names, inputs)

        # Flatten the output and store it as the encoder output
        self.encoder_out = np.array(output).flatten()
 
    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        This method updates the joint's desired position, velocity, torque, and control gains.
        Replace this implementation with the actual communication logic for your hardware.

        Parameters:
        joint_index (int): The index of the joint to be controlled.
        q (float): The desired joint position, typically in radians or degrees.
        dq (float): The desired joint velocity, typically in radians/second or degrees/second.
        tau (float): The desired joint torque, typically in Newton-meters (Nm).
        kp (float): The proportional gain for position control.
        kd (float): The derivative gain for velocity control.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """
        Updates the robot's state based on the current mode and publishes the robot command.
        """
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        # Increment the loop count
        self.loop_count += 1

        # Publish the robot command
        self.robot.publishRobotCmd(self.robot_cmd)
        
    # Callback function for receiving robot command data
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        Callback function to update the robot state from incoming data.
        
        Parameters:
        robot_state (datatypes.RobotState): The current state of the robot.
        """
        self.robot_state = robot_state

    # Callback function for receiving imu data
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        Callback function to update IMU data from incoming data.
        
        Parameters:
        imu_data (datatypes.ImuData): The IMU data containing stamp, acceleration, gyro, and quaternion.
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # Rotate quaternion values
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    # Callback function for receiving sensor joy data
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # Check if the robot is in the calibration state and both L1 (button index 4) and Y (button index 3) buttons are pressed.
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
          print(f"L1 + Y: start_controller...")
          self.start_controller = True

        # Check if both L1 (button index 4) and X (button index 2) are pressed to stop the controller
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
          print(f"L1 + X: stop_controller...")
          self.start_controller = False

        linear_x  = sensor_joy.axes[1]
        linear_y  = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        linear_x  = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y  = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        if self.is_mjlab:
            trimmed_commands = np.array([linear_x, linear_y, angular_z]) + self.mjlab_command_offsets
            trimmed_commands = np.clip(trimmed_commands, -1.0, 1.0)
            trimmed_commands = np.array([
                self.apply_deadband(trimmed_commands[0], self.mjlab_command_deadband),
                self.apply_deadband(trimmed_commands[1], self.mjlab_command_deadband),
                self.apply_deadband(trimmed_commands[2], self.mjlab_command_deadband),
            ])
            self.commands[:] = trimmed_commands
            return

        self.commands[0] = linear_x
        self.commands[1] = linear_y
        self.commands[2] = angular_z

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      # Check if the received diagnostic data is related to calibration.
      if diagnostic_value.name == "calibration":
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code
