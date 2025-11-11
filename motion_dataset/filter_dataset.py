import joblib
import yaml

def amass2twist(amass_names, twist_names, amass_name, twist_name, dir_level=-1):
    amass_extra, amass_reserved = [], []
    t_names = [name for name in twist_names if name.split('/')[0] == twist_name]
    a_names = [name for name in amass_names if name.split('/')[0] == amass_name]
    for a_name in a_names:
        if amass_name in ['GRAB', 'CNRS']:
            a2t_name = f'{twist_name}/' + '_'.join(a_name.split('/')[dir_level:]).replace('-', '_').replace('(', '_').replace(')', '_')
        elif amass_name in ['BMLhandball']:
            a2t_name = f'{twist_name}/' + '_'.join(a_name.split('/')[dir_level:]).replace('-', '_').replace('_poses', '').replace('(', '_').replace(')', '_')
        else:
            a2t_name = f'{twist_name}/' + '_'.join(a_name.split('/')[dir_level:]).replace('-', '_').replace('_stageii', '').replace('(', '_').replace(')', '_')
        if a2t_name not in t_names:
            amass_extra.append(a_name)
        else:
            amass_reserved.append(a_name)
            t_names.remove(a2t_name)
    return amass_extra, t_names, amass_reserved

if __name__ == '__main__':
    # from omegaconf import OmegaConf
    # orca_config = OmegaConf.load('legged_gym/motion_data_configs/motion_dataset.yaml')
    # orca_files = [motion.file for motion in orca_config.motions]
    # twist_config = OmegaConf.load('legged_gym/motion_data_configs/twist_dataset.yaml')
    # twist_files = [motion.file for motion in twist_config.motions]
    # joblib.dump({'twist_files': twist_files, 'orca_files': orca_files}, 'motion_files.pkl')
    # exit()

    motion_files = joblib.load('motion_files.pkl')
    twist_files = motion_files['twist_files']
    orca_files = motion_files['orca_files']
    amass_extra_files, twist_extra_files, amass_reserved = [], [], []
    # names = set()
    # for n in twist_files:
    #     names.add(n.split('/')[0])
    # print(names)
    # exit()

    dataset_list = [
        ('ACCAD', 'accad', -1),
        ('Transitions', 'transitions', -1),
        ('PosePrior', 'mpi_limits', -2),
        ('Eyes_Japan_Dataset', 'eyes_japan', -1),
        ('MoSh', 'mpi_mosh', -2),
        ('HUMAN4D', 'human4d', -2),
        ('GRAB', 'grab', -2),
        ('HDM05', 'hdm05', -1),
        ('DanceDB', 'dancedb', -1),
        ('CNRS', 'cnrs', -2),
        ('SFU', 'sfu', -1),
        ('EKUT', 'ekut', -2),
        ('KIT', 'kit', -1),
        ('DFaust', 'dfaust', -1),
        ('HumanEva', 'humaneva', -2),
        ('TotalCapture', 'totalcapture', -2),
        ('BMLhandball', 'bmlhandball', -2),
        ('CMU', 'cmu', -1),
        ('BMLmovi', 'bmlmovi', -1),
        ('omomo', 'omomo', -1),
        ]

    for amass_name, twist_name, dir_level in dataset_list:
        a, t, r = amass2twist(orca_files, twist_files, amass_name=amass_name, twist_name=twist_name, dir_level=dir_level)
        amass_extra_files.extend(a)
        twist_extra_files.extend(t)
        amass_reserved.extend(r)
        print(f'--------------extra twist: {twist_name} -------------')
        print(len(t), len(a), len(r))
        # for name in a:
        #     print(name)
        print('---------------------------')
        print(f'Generate new motion dataset config for {len(amass_reserved)} data ...')
    config = {'root_path': 'motion_dataset',
                'motions': []}
    for name in sorted(amass_reserved):
        motion_info = {
            'file': name,
            'weight': 1.0,
            'description': 'general movement'
        }
        config['motions'].append(motion_info)
    with open('motion_dataset.yaml', 'w') as f:
        yaml.dump(config, f)
