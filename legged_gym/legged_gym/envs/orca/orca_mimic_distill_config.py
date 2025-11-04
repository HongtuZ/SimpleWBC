from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class OrcaMimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 28
        num_key_bodies = 12
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3 # 没用？
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions #本体感知观测
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*num_key_bodies) # Hardcode for now, 8 is base
        n_mimic_obs = 8 + 28 # 28 for dof pos
        n_priv_info = 3 + 1 + 3*num_key_bodies + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single

        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        track_root = False
     
        dof_err_w = [1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Left Leg
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Right Leg
                     0.6, 0,0,0,# waist yaw, head, 
                     0.8, 0.8, 0.8, 1.0,1,1, # Left Arm
                     0.8, 0.8, 0.8, 1.0,1,1,# Right Arm
                     ]
        

        
        global_obs = False
        # global_obs = True
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            'waist_yaw_joint': 0.0,

            'larm_joint1': 0.0,
            'larm_joint2': 0.0,
            'larm_joint3': 0.0,
            'larm_joint4': 0.0,
            'larm_joint5': 0.0,
            'larm_joint6': 0.0,
            
            'rarm_joint1': 0.0,
            'rarm_joint2': 0.0,
            'rarm_joint3': 0.0,
            'rarm_joint4': 0.0,
            'rarm_joint5': 0.0,
            'rarm_joint6': 0.0,
            
            'head_joint1':0.0,
            'head_joint2':0.0,
            'head_joint3':0.0,

            'lleg_joint1': 0.0,
            'lleg_joint2': 0.0,
            'lleg_joint3': 0.0,
            'lleg_joint4': 0.0,
            'lleg_joint5': 0.0,
            'lleg_joint6': 0.0,
            
            'rleg_joint1': 0.0,
            'rleg_joint2': 0.0,
            'rleg_joint3': 0.0,
            'rleg_joint4': 0.0,
            'rleg_joint5': 0.0,
            'rleg_joint6': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {  
                        'arm_joint1': 100.,
                        'arm_joint2': 100.,
                        'arm_joint3': 100.,
                        'arm_joint4': 100.,
                        'arm_joint5': 100.,
                        'arm_joint6': 100.,
                        'leg_joint1': 75.,
                        'leg_joint2': 50.,
                        'leg_joint3': 50.,
                        'leg_joint4': 75.,
                        'leg_joint5': 30.,
                        'leg_joint6': 5, #0.,
                        'waist': 50
                    }  # [N*m/rad]

        damping =   {  
                        'arm_joint1': 3.,
                        'arm_joint2': 3.,
                        'arm_joint3': 3.,
                        'arm_joint4': 3.,
                        'arm_joint5': 2.,
                        'arm_joint6': 1.,
                        'leg_joint1': 3.,
                        'leg_joint2': 3.,
                        'leg_joint3': 3., # 5.
                        'leg_joint4': 3.,
                        'leg_joint5': 2.,
                        'leg_joint6': 1, #5.,
                        "waist": 1
                    }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002 # 1/500
        # dt = 1/200 # 0.005
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        # file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision.urdf'
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/orca/urdf/orca.urdf'
        
        # for both joint and link name
        torso_name: str = 'base_link'  # humanoid pelvis part
        chest_name: str = 'waist_yaw_link1'  # humanoid chest part

        # for link name
        thigh_name: str = ['leg_link1','leg_link2','leg_link3']
        shank_name: str = 'leg_link4'
        foot_name: str = 'leg_link6'  # foot_pitch is not used
        waist_name: list = ['waist_yaw_link1']
        upper_arm_name: str = 'arm_link2'
        lower_arm_name: str = 'arm_link4'
        hand_name: list = ['rarm_link6', 'larm_link6']

        feet_bodies = ['lleg_link6', 'rleg_link6']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["arm_link1","arm_link2","arm_link3","arm_link4","leg_link1","leg_link2","leg_link3","leg_link4"]
        # terminate_after_contacts_on = ['waist_yaw_link1']
        terminate_after_contacts_on = ['base_link']
        
        # ========================= Inertia =========================
        # shoulder, elbow, and ankle: 0.139 * 1e-4 * 16**2 + 0.017 * 1e-4 * (46/18 + 1)**2 + 0.169 * 1e-4 = 0.003597
        # waist, hip pitch & yaw: 0.489 * 1e-4 * 14.3**2 + 0.098 * 1e-4 * 4.5**2 + 0.533 * 1e-4 = 0.0103
        # knee, hip roll: 0.489 * 1e-4 * 22.5**2 + 0.109 * 1e-4 * 4.5**2 + 0.738 * 1e-4 = 0.0251
        # wrist: 0.068 * 1e-4 * 25**2 = 0.00425
        
        # dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 4 + [0.003597] * 12
        
        # dof_armature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001] * 2 + [0.0] * 3 + [0.0] * 8
        
        # ========================= Inertia =========================
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.0
            
            # alive = 0.5

            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            # feet_height = 5.0
            feet_air_time = 5.0
            
            
            ang_vel_xy = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        # push_end_effector = False
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        #key_bodies = ["larm_link6", "rarm_link6", "lleg_link6", "rleg_link6", "lleg_link4", "rleg_link4", "larm_link4", "rarm_link4", "head_link3"] # 9 key bodies
        #upper_key_bodies = ["larm_link6", "rarm_link6", "larm_link4", "rarm_link4", "head_link3"]
        key_bodies = ['larm_link2', 'larm_link4', 'larm_link6', 'rarm_link2', 'rarm_link4', 'rarm_link6', 'lleg_link1', 'lleg_link4', 'lleg_link6', 'rleg_link1', 'rleg_link4', 'rleg_link6'] # 12 key bodies
        upper_key_bodies = ['larm_link2', 'larm_link4', 'larm_link6', 'rarm_link2', 'rarm_link4', 'rarm_link6']
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data/0018_Catwalk001_stageii.pkl"
        # motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/motion_dataset.yaml"
        reset_consec_frames = 30
        height_offset = 0.1
    

class OrcaMimicStuCfg(OrcaMimicPrivCfg):
    class env(OrcaMimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 28
        num_key_bodies = 12
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*num_key_bodies) # Hardcode for now, 12 is the number of key bodies
        n_mimic_obs = 8 + 28 # 23 for dof pos

        n_priv_info = 3 + 1 + 3*num_key_bodies + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

class OrcaMimicStuRLCfg(OrcaMimicPrivCfg):
    class env(OrcaMimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 28
        num_key_bodies = 12
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*num_key_bodies) # Hardcode for now, 12 is the number of key bodies
        n_mimic_obs = 8 + 28 # 23 for dof pos
        
        n_priv_info = 3 + 1 + 3*num_key_bodies + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.0
            
            alive = 0.5

            feet_slip = -0.1 # higher than teacher
            feet_contact_forces = -5e-4      
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            feet_air_time = 5.0
            
            
            ang_vel_xy = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

class OrcaMimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 1_000_002 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        # Transformer params
        # learning_rate = 1e-4 #1.e-3 #5.e-4
        # schedule = 'fixed' # could be adaptive, fixed
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.4] + [0.5] * 12 + [0.4] * 3 + [0.7] * 12
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        


class OrcaMimicStuRLCfgDAgger(OrcaMimicStuRLCfg):
    seed = 1
    
    class teachercfg(OrcaMimicPrivCfgPPO):
        pass
    
    class runner(OrcaMimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'orca_tacher'
        teacher_proj_name = 'orca_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        
        dagger_coef = 0.1
        dagger_coef_min = 0.01  # Minimum value for dagger_coef
        # dagger_coef = 0.0
        # dagger_coef_min = 0.0  # Minimum value for dagger_coef

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.4] + [0.5] * 12 + [0.4] * 3 + [0.7] * 12
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128