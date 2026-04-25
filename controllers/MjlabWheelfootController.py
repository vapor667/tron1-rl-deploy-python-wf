import copy
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

from .WheelfootController import WheelfootController


class MjlabWheelfootController(WheelfootController):
    def __init__(self, model_dir, robot, robot_type, rl_type, start_controller):
        if rl_type != "mjlab":
            raise ValueError(
                f"MjlabWheelfootController only supports rl_type='mjlab', got '{rl_type}'"
            )
        super().__init__(model_dir, robot, robot_type, rl_type, start_controller)

    def initialize_onnx_models(self):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False

        cpu_providers = ["CPUExecutionProvider"]
        self.policy_session = ort.InferenceSession(
            self.model_policy, sess_options=session_options, providers=cpu_providers
        )
        self.policy_input_names = [
            self.policy_session.get_inputs()[i].name
            for i in range(self.policy_session.get_inputs().__len__())
        ]
        self.policy_output_names = [
            self.policy_session.get_outputs()[i].name
            for i in range(self.policy_session.get_outputs().__len__())
        ]
        self.policy_input_shapes = [
            self.policy_session.get_inputs()[i].shape
            for i in range(self.policy_session.get_inputs().__len__())
        ]
        self.policy_output_shapes = [
            self.policy_session.get_outputs()[i].shape
            for i in range(self.policy_session.get_outputs().__len__())
        ]

        self.encoder_session = None
        self.encoder_input_names = []
        self.encoder_output_names = []
        self.encoder_input_shapes = []
        self.encoder_output_shapes = []
        self.validate_mjlab_policy()

    def validate_mjlab_policy(self):
        self.policy_obs_history_size = None
        expected_shapes = {
            "obs": [1, self.observations_size],
            "commands": [1, self.mjlab_policy_commands_size],
        }
        if len(self.policy_input_names) != 3:
            raise ValueError(f"mjlab policy expects 3 inputs, got {self.policy_input_names}")

        for input_name, input_shape in zip(self.policy_input_names, self.policy_input_shapes):
            if input_name == "obs_history":
                if len(input_shape) != 2 or input_shape[0] != 1:
                    raise ValueError(
                        f"mjlab policy input 'obs_history' shape {input_shape} must be [1, N]"
                    )
                self.policy_obs_history_size = int(input_shape[1])
                continue

            if input_name not in expected_shapes:
                raise ValueError(f"Unexpected mjlab policy input '{input_name}'")
            if list(input_shape) != expected_shapes[input_name]:
                raise ValueError(
                    f"mjlab policy input '{input_name}' shape {input_shape} does not match expected {expected_shapes[input_name]}"
                )

        if self.policy_obs_history_size is None:
            raise ValueError("mjlab policy missing required input 'obs_history'")

        supported_obs_history_sizes = {
            self.observations_size,
            self.obs_history_length * self.observations_size,
        }
        if self.policy_obs_history_size not in supported_obs_history_sizes:
            raise ValueError(
                f"Unsupported mjlab policy obs_history size {self.policy_obs_history_size}; "
                f"expected one of {sorted(supported_obs_history_sizes)}"
            )

        if len(self.policy_output_shapes) != 1 or list(self.policy_output_shapes[0]) != [1, self.actions_size]:
            raise ValueError(
                f"mjlab policy output shape {self.policy_output_shapes} does not match expected [[1, {self.actions_size}]]"
            )

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        pointfoot_cfg = config["PointfootCfg"]
        mjlab_cfg = pointfoot_cfg.get("mjlab", {})
        size_cfg = copy.deepcopy(pointfoot_cfg["size"])
        size_cfg.update(mjlab_cfg.get("size", {}))

        self.control_cfg = copy.deepcopy(pointfoot_cfg["control"])
        self.control_cfg.update(mjlab_cfg.get("control", {}))

        self.joint_names = pointfoot_cfg["joint_names"]
        self.init_state = pointfoot_cfg["init_state"]["default_joint_angle"]
        self.stand_duration = pointfoot_cfg["stand_mode"]["stand_duration"]
        self.rl_cfg = pointfoot_cfg["normalization"]
        self.obs_scales = self.rl_cfg["obs_scales"]
        self.actions_size = size_cfg["actions_size"]
        self.commands_size = size_cfg["commands_size"]
        self.observations_size = size_cfg["observations_size"]
        self.obs_history_length = size_cfg["obs_history_length"]
        self.encoder_output_size = size_cfg["encoder_output_size"]
        self.imu_orientation_offset = np.array(list(pointfoot_cfg["imu_orientation_offset"].values()))
        self.user_cmd_cfg = pointfoot_cfg["user_cmd_scales"]
        self.user_cmd_offsets = pointfoot_cfg.get(
            "user_cmd_offsets",
            {
                "lin_vel_x": 0.0,
                "lin_vel_y": 0.0,
                "ang_vel_yaw": 0.0,
            },
        )
        self.loop_frequency = pointfoot_cfg["loop_frequency"]
        self.encoder_input_size = self.obs_history_length * self.observations_size

        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.command_actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)
        self.scaled_commands = np.zeros(self.commands_size)
        self.base_lin_vel = np.zeros(3)
        self.base_position = np.zeros(3)
        self.base_pose_commands = np.zeros(4)
        self.base_se3_decrease_rate = np.zeros(1)
        self.mjlab_history_buffers = []
        self.loop_count = 0
        self.stand_percent = 0
        self.policy_session = None
        self.joint_num = len(self.joint_names)

        self.joint_pos_idxs = size_cfg["jointpos_idxs"]
        self.wheel_joint_damping = self.control_cfg["wheel_joint_damping"]
        self.wheel_joint_torque_limit = self.control_cfg["wheel_joint_torque_limit"]
        self.action_scale_vel = self.control_cfg.get("action_scale_vel", 5.0)

        mjlab_command_cfg = mjlab_cfg.get("command", {})
        self.mjlab_base_pose_scale = mjlab_command_cfg.get("base_pose_scale_xy", 2.0)
        self.mjlab_base_se3_decrease_rate = mjlab_command_cfg.get("base_se3_decrease_rate", 1.0)
        self.mjlab_target_orient_x_axis = np.array(
            mjlab_command_cfg.get("target_orient_x_axis", [1.0, 0.0]),
            dtype=np.float32,
        )
        self.mjlab_command_offsets = np.array(
            [
                mjlab_command_cfg.get("input_offsets", {}).get(
                    "lin_vel_x", self.user_cmd_offsets["lin_vel_x"]
                ),
                mjlab_command_cfg.get("input_offsets", {}).get(
                    "lin_vel_y", self.user_cmd_offsets["lin_vel_y"]
                ),
                mjlab_command_cfg.get("input_offsets", {}).get(
                    "ang_vel_yaw", self.user_cmd_offsets["ang_vel_yaw"]
                ),
            ]
        )
        self.mjlab_cmd_offsets = np.array(
            [
                mjlab_command_cfg.get("cmd_offsets", {}).get("lin_vel_x", 0.0),
                mjlab_command_cfg.get("cmd_offsets", {}).get("lin_vel_y", 0.0),
                mjlab_command_cfg.get("cmd_offsets", {}).get("ang_vel_yaw", 0.0),
            ]
        )
        self.mjlab_command_deadband = mjlab_command_cfg.get("input_deadband", 0.05)
        self.mjlab_policy_commands_size = mjlab_command_cfg.get("policy_commands_size", 0)
        self.mjlab_history_term_dims = mjlab_command_cfg.get(
            "history_term_dims", [4, 1, 3, 3, 3, 6, 8, 8]
        )
        if sum(self.mjlab_history_term_dims) != self.observations_size:
            raise ValueError(
                f"mjlab history term dims {self.mjlab_history_term_dims} do not sum to observations_size {self.observations_size}"
            )
        self.policy_commands = np.zeros(self.mjlab_policy_commands_size)

        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]

        self.mode = "STAND"

    def handle_walk_mode(self):
        self.init_joint_angles[1] = 0.0
        self.init_joint_angles[5] = 0.0

        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        if self.loop_count % self.control_cfg["decimation"] == 0:
            self.compute_observation()
            self.compute_encoder()
            self.compute_actions()
            action_min = -self.rl_cfg["clip_scales"]["clip_actions"]
            action_max = self.rl_cfg["clip_scales"]["clip_actions"]
            self.actions = np.clip(self.actions, action_min, action_max)
            self.last_actions = np.array(self.actions)
            self.command_actions = self.reorder_mjlab_actions_to_robot(self.actions)

        command_actions = np.array(self.command_actions)
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:
                action_min = (
                    joint_pos[i]
                    - self.init_joint_angles[i]
                    + (
                        self.control_cfg["damping"] * joint_vel[i]
                        - self.control_cfg["user_torque_limit"]
                    )
                    / self.control_cfg["stiffness"]
                )
                action_max = (
                    joint_pos[i]
                    - self.init_joint_angles[i]
                    + (
                        self.control_cfg["damping"] * joint_vel[i]
                        + self.control_cfg["user_torque_limit"]
                    )
                    / self.control_cfg["stiffness"]
                )
                action_value = max(
                    action_min / self.control_cfg["action_scale_pos"],
                    min(action_max / self.control_cfg["action_scale_pos"], command_actions[i]),
                )
                pos_des = action_value * self.control_cfg["action_scale_pos"] + self.init_joint_angles[i]
                self.set_joint_command(
                    i, pos_des, 0, 0, self.control_cfg["stiffness"], self.control_cfg["damping"]
                )
            else:
                velocity_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                velocity_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                velocity_des = np.clip(
                    command_actions[i] * self.action_scale_vel, velocity_min, velocity_max
                )
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def reorder_mjlab_actions_to_robot(self, policy_actions):
        robot_actions = np.zeros(policy_actions.shape, dtype=policy_actions.dtype)
        robot_actions[0:3] = policy_actions[0:3]
        robot_actions[3] = policy_actions[6]
        robot_actions[4:7] = policy_actions[3:6]
        robot_actions[7] = policy_actions[7]
        return robot_actions

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

        self.proprio_history_vector = np.concatenate(
            [buffer.reshape(-1) for buffer in self.mjlab_history_buffers]
        )
        self.proprio_history_buffer = np.array(self.proprio_history_vector)

    def compute_observation(self):
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler("zyx")
        inverse_rot = R.from_euler("zyx", q_wi).inv().as_matrix()

        gravity_vector = np.array([0, 0, -1])
        projected_gravity = np.dot(inverse_rot, gravity_vector)

        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        rot = R.from_euler("zyx", self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, base_ang_vel)
        projected_gravity = np.dot(rot, projected_gravity)

        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)
        actions = np.array(self.last_actions)

        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales["dof_pos"]
        joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])

        effective_commands = np.clip(self.commands + self.mjlab_cmd_offsets, -1.0, 1.0)
        self.base_pose_commands = np.array(
            [
                effective_commands[0] * self.mjlab_base_pose_scale,
                effective_commands[1] * self.mjlab_base_pose_scale,
                self.mjlab_target_orient_x_axis[0],
                self.mjlab_target_orient_x_axis[1],
            ]
        )
        self.base_se3_decrease_rate = np.array([self.mjlab_base_se3_decrease_rate])
        self.scaled_commands = np.array(effective_commands)

        obs_terms = [
            self.base_pose_commands,
            self.base_se3_decrease_rate,
            self.scaled_commands,
            base_ang_vel * self.obs_scales["ang_vel"],
            projected_gravity,
            joint_pos_input,
            joint_velocities * self.obs_scales["dof_vel"],
            actions,
        ]
        obs = np.concatenate(obs_terms)
        self.update_mjlab_history(obs_terms)
        self.observations = np.clip(
            obs,
            -self.rl_cfg["clip_scales"]["clip_observations"],
            self.rl_cfg["clip_scales"]["clip_observations"],
        )

    def get_policy_obs_history_input(self):
        # New fixed student path: obs_history is actor observation (same size as obs).
        if self.policy_obs_history_size == self.observations_size:
            return self.observations

        # Backward compatibility for old exported models.
        return self.proprio_history_vector

    def compute_actions(self):
        if self.policy_session is None:
            raise RuntimeError("Policy session is not initialized")

        inputs = {
            "obs_history": self.get_policy_obs_history_input().astype(np.float32).reshape(1, -1),
            "obs": self.observations.astype(np.float32).reshape(1, -1),
            "commands": self.policy_commands.astype(np.float32).reshape(1, -1),
        }
        output = self.policy_session.run(self.policy_output_names, inputs)
        self.actions = np.array(output[0]).flatten()

    def compute_encoder(self):
        return

    def sensor_joy_callback(self, sensor_joy):
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
            print("L1 + Y: start_controller...")
            self.start_controller = True

        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
            print("L1 + X: stop_controller...")
            self.start_controller = False

        linear_x = sensor_joy.axes[1]
        linear_y = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        linear_x = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        trimmed_commands = np.array([linear_x, linear_y, angular_z]) + self.mjlab_command_offsets
        trimmed_commands = np.clip(trimmed_commands, -1.0, 1.0)
        trimmed_commands = np.array(
            [
                self.apply_deadband(trimmed_commands[0], self.mjlab_command_deadband),
                self.apply_deadband(trimmed_commands[1], self.mjlab_command_deadband),
                self.apply_deadband(trimmed_commands[2], self.mjlab_command_deadband),
            ]
        )
        self.commands[:] = trimmed_commands
