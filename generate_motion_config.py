"""
Codes based on
https://github.com/TimoStoff/esim_config_generator
@Article{Stoffregen20eccv,
  author        = {T. Stoffregen, C. Scheerlinck, D. Scaramuzza, T. Drummond, N. Barnes, L. Kleeman, R. Mahoney},
  title         = {Reducing the Sim-to-Real Gap for Event Cameras},
  journal       = eccv,
  year          = 2020,
  month         = aug
}
"""
import argparse
import numpy as np
import json
import random
import os
import glob
from collections import OrderedDict

PI = 3.1415926

def read_json(fname):
    assert(os.path.exists(fname))
    with open(fname) as json_file:
        data = json.load(json_file, object_hook=OrderedDict)
        return data

def get_random_translations(speed, duration, image_size):
    """
    Get motion vectors through a rectangle of width 1, centered
    at 0, 0 (the virtual image plane), which have a trajectory
    length such that the given speed of the motion occurs, given
    the duration of the sequence. For example, if a very fast motion
    is desired for a long scene, the trajectory is going to be have
    to be huge to respect the desired speed.
    """
    displacement = np.array([speed*duration])/image_size
    edge_limit = 0.45
    rnd_point_in_image = np.array((random.uniform(-edge_limit, edge_limit), random.uniform(-edge_limit, edge_limit)))
    rnd_direction = np.array(random.uniform(0, 2*PI))
    if np.linalg.norm(displacement) < 3.0:
        rnd_speed_component = np.array(random.uniform(0.5, 0.5)*displacement)
    else:
        rnd_speed_component = np.array(random.uniform(0.1, 0.9) * displacement)
    vec = np.array([np.cos(rnd_direction), np.sin(rnd_direction)])

    point_1 = (vec * rnd_speed_component) + rnd_point_in_image
    point_2 = (-vec * (displacement-rnd_speed_component)) + rnd_point_in_image

    return point_1[0], point_1[1], point_2[0], point_2[1]


def get_motion_string(image_name, duration, image_size, object_speed, min_ang_vel_deg, max_ang_vel_deg, min_growth, max_growth, mfs=3, gbs=0):
    """
    Given an image and the desired trajectory, return a string that the simulator can read
    """
    x0, y0, x1, y1 = get_random_translations(object_speed, duration, image_size)

    random_angular_v = random.uniform(min_ang_vel_deg, max_ang_vel_deg) * (1 if random.random() < 0.5 else -1)
    angle_to_travel = random_angular_v * duration
    random_start_angle = random.uniform(0, 360)
    theta0 = random_start_angle
    theta1 = random_start_angle + angle_to_travel

    sx0 = random.uniform(min_growth, max_growth)
    sx1 = random.uniform(min_growth, max_growth)
    sy0 = random.uniform(min_growth, max_growth)
    sy1 = random.uniform(min_growth, max_growth)
    return "{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
        image_name, mfs, gbs, theta0, theta1, x0, x1, y0, y1, sx0, sx1, sy0, sy1), [x0, y0, x1, y1]


def generate_scene_file_multispeed(output_path, image_size, duration, background_images, foreground_images,
                                   number_objects=15, median_filter_size=3, gaussian_blur_sigma=0,
                                   bg_min_ang_velocity_deg=0, bg_max_ang_velocity_deg=5, bg_min_velocity=5,
                                   bg_max_velocity=20, bg_growthmin=1.5, bg_growthmax=2.0,
                                   fg_min_ang_velocity_deg=0, fg_max_ang_velocity_deg=400, fg_min_velocity=5,
                                   fg_max_velocity=500, fg_growthmin=0.5, fg_growthmax=1.5, proportion_variation=0.1,
                                   fully_random=True):
    """
    Given foreground and background image directories and motion parameters, sample some random images
    and give them velocities distributed across the range of velocities given. Velocities are sampled
    linearly across the range, with a small percentage variation, as to guarantee the full 'spread' of motions.
    Otherwise, if you don't want that behaviour, you may set 'fully_random' to True, in which case the speeds
    will be chosen from a uniform distribution between fg_min_velocity and fg_max_velocity.
    """
    f = open(output_path, "w")
    f.write("{} {} {}\n".format(image_size[0], image_size[1], duration))

    background_image_paths = []
    for image_set in background_images:
        background_image_paths.extend(sorted(glob.glob("{}/*.jpg".format(background_images[image_set]['path']))))
    assert(len(background_image_paths) > 0)
    background = random.choice(background_image_paths)

    foreground_image_paths = []
    if foreground_images['min_num'] == foreground_images['max_num']:
        num_images = 1
    else:
        num_images = np.random.randint(foreground_images['min_num'], foreground_images['max_num'])
    print("{} foreground images".format(num_images))
    for image_set in foreground_images:
        if isinstance(foreground_images[image_set], dict):
            all_paths = sorted(glob.glob("{}/*.png".format(foreground_images[image_set]['path'])))
            all_paths.extend(glob.glob("{}/*.jpg".format(foreground_images[image_set]['path'])))
            # print(foreground_images[image_set], int(num_images*foreground_images[image_set]['proportion']+0.5))
            selection = random.sample(all_paths, int(num_images*foreground_images[image_set]['proportion']+0.5))
            foreground_image_paths.extend(selection)
    assert(len(foreground_image_paths) > 0)
    random.shuffle(foreground_image_paths)

    #Background
    random_speed = random.uniform(bg_min_velocity, bg_max_velocity)
    f.write(get_motion_string(background, duration, image_size, random_speed, bg_min_ang_velocity_deg, bg_max_ang_velocity_deg,
                              bg_growthmin, bg_growthmax, median_filter_size, gaussian_blur_sigma)[0])
    #Foreground
    object_speeds = []
    obj_strings = []
    v_range = np.linspace(fg_min_velocity, fg_max_velocity, len(foreground_image_paths))
    for i, fg in enumerate(foreground_image_paths):
        if fully_random:
            v_vel = np.random.normal(fg_min_velocity, fg_max_velocity)
        else:
            v_vel = v_range[i]*np.random.normal(1, proportion_variation)

        m_string, motion = get_motion_string(fg, duration, image_size, v_vel, fg_min_ang_velocity_deg,
                                             fg_max_ang_velocity_deg, fg_growthmin, fg_growthmax,
                                             median_filter_size, gaussian_blur_sigma)
        object_speeds.append(motion)
        obj_strings.append(m_string)

    random.shuffle(obj_strings)
    for obj_string in obj_strings:
        f.write(obj_string)

    f.close
    print("Wrote new scene file to {}".format(output_path))
    return object_speeds


def generate_background_scene_file_multispeed(output_path, image_size, duration, background_images,
                                   number_objects=15, median_filter_size=3, gaussian_blur_sigma=0,
                                   bg_min_ang_velocity_deg=0, bg_max_ang_velocity_deg=5, bg_min_velocity=5,
                                   bg_max_velocity=20, bg_growthmin=1.5, bg_growthmax=2.0,
                                   ):
    """
    Background only
    Given background image directories and motion parameters, sample some random images
    and give them velocities distributed across the range of velocities given. Velocities are sampled
    linearly across the range, with a small percentage variation, as to guarantee the full 'spread' of motions.
    Otherwise, if you don't want that behaviour, you may set 'fully_random' to True, in which case the speeds
    will be chosen from a uniform distribution between fg_min_velocity and fg_max_velocity.
    """
    f = open(output_path, "w")
    f.write("{} {} {}\n".format(image_size[0], image_size[1], duration))

    background_image_paths = []
    for image_set in background_images:
        background_image_paths.extend(sorted(glob.glob("{}/*.jpg".format(background_images[image_set]['path']))))
    assert(len(background_image_paths) > 0)
    background = random.choice(background_image_paths)

    #Background
    random_speed = random.uniform(bg_min_velocity, bg_max_velocity)
    f.write(get_motion_string(background, duration, image_size, random_speed, bg_min_ang_velocity_deg, bg_max_ang_velocity_deg,
                              bg_growthmin, bg_growthmax, median_filter_size, gaussian_blur_sigma)[0])
    f.close
    print("Wrote new scene file to {}".format(output_path))
    return random_speed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scene file generator')
    parser.add_argument('--generator_config', type=str, help='Scene generator settings',
            default=None)
    parser.add_argument('--output_path', type=str, help='Path to save scene file', required=True) # '/Users/liusiying/research/data/COCO/configs'
    parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate', default=600)
    parser.add_argument('--seq_start_id', type=int, help='The start ID of sequences to save', default=0)
    
    #Scene params
    parser.add_argument('--image_width', type=int, help='Image width (pixels)', default=240)
    parser.add_argument('--image_height', type=int, help='Image height (pixels)', default=180)
    parser.add_argument('--scene_duration', type=float, help='How long should the sequence go\
            (seconds)', default=2.0)
    
    parser.add_argument('--is_bg_only', action='store_true', help='True if the scene only has background', default=False)


    args = parser.parse_args()
    start_id = args.seq_start_id

    output_path = args.output_path
    image_size = (args.image_width, args.image_height)
    duration = args.scene_duration
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if args.generator_config is None:
        config_json_folder = 'motion_config'
        for file_name in os.listdir(config_json_folder):
            if file_name.split('.')[-1] == 'json':
                args.generator_config = os.path.join(config_json_folder, file_name)
                # /variety/fast/medium/slow
                config = read_json(args.generator_config)

                #Scene generation
                motion_params = config['foreground_params']
                motion_params.update(config['background_params'])
                
                for id in range(start_id, start_id+args.num_sequences):
                    # path_to_config_file = os.path.join(output_path, '{}_{:09d}_autoscene.txt'.format(file_name.split('/')[-1].split('.')[0], id))
                    path_to_config_file = os.path.join(output_path, 'sequence_{:010d}.txt'.format(id))
                    motions = generate_scene_file_multispeed(path_to_config_file, image_size, duration,
                            config['background_images'], config['foreground_images'], **motion_params)
                start_id += args.num_sequences

    else:
        config = read_json(args.generator_config)

        if 'foreground_params' in config and (not args.is_bg_only):
            motion_params = config['foreground_params']
            motion_params.update(config['background_params'])
            
            for id in range(start_id, start_id+args.num_sequences):
                # path_to_config_file = os.path.join(output_path, '{}_{:09d}_autoscene.txt'.format(file_name.split('/')[-1].split('.')[0], id))
                path_to_config_file = os.path.join(output_path, 'sequence_{:010d}.txt'.format(id))
                motions = generate_scene_file_multispeed(path_to_config_file, image_size, duration,
                        config['background_images'], config['foreground_images'], **motion_params)
        else:
            # background only
            motion_params = config['background_params'] 
            
            for id in range(start_id, start_id+args.num_sequences):
                # path_to_config_file = os.path.join(output_path, '{}_{:09d}_autoscene.txt'.format(file_name.split('/')[-1].split('.')[0], id))
                path_to_config_file = os.path.join(output_path, 'sequence_{:010d}.txt'.format(id))
                motions = generate_background_scene_file_multispeed(path_to_config_file, image_size, duration,
                        config['background_images'], **motion_params)




