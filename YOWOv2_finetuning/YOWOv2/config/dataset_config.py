# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'gt_folder': './evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': False,  # one hot
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 24,
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
    },
    
    'ava_v2.2':{
        # dataset
        'frames_dir': '/content/AVA_Dataset/frames/',
        'frame_list': '/content/AVA_Dataset/frame_lists/',
        'annotation_dir': '/content/AVA_Dataset/annotations/',
        'train_gt_box_list': 'ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_action_list_v2.2.pbtxt', # 'ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt', 
        #'class_ratio_file': 'config/ava_categories_ratio.json',
        'backup_dir': 'results/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': True,  # multi hot
        # train config
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 20, #500
        # class names
        'valid_num_classes': 15,
        'label_map': (
                    "baseball_pitch", "baseball_swing", "golf_swing", "squat",
          "jumping_jacks", "clean_and_jerk", "bench_press", "tennis_serve","pushup", 
          "situp", "pullup", "bowl", "jump_rope", "strum_guitar", "tennis_forehand"
        ),
    }
}