import glob
import argparse
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np
import csv
import zipfile


# https://github.com/TimoStoff/events_contrast_maximization/blob/d6241dc90ec4dc2b4cffbb331a2389ff179bf7ab/tools/rosbag_to_h5.py
# rosbag_to_h5

def append_to_dataset(dataset, data):
    dataset.resize(dataset.shape[0] + len(data), axis=0)
    if len(data) == 0:
        return
    dataset[-len(data):] = data[:]


def timestamp_float(ts):
    return ts.secs + ts.nsecs / float(1e9)


def get_rosbag_stats(bag, event_topic=None, image_topic=None, flow_topic=None):
    num_event_msgs = 0
    num_img_msgs = 0
    num_flow_msgs = 0
    topics = bag.get_type_and_topic_info().topics
    # print(topics)
    for topic_name, topic_info in topics.items():
        if topic_name == event_topic:
            num_event_msgs = topic_info.message_count
            print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))
        if topic_name == image_topic:
            num_img_msgs = topic_info.message_count
            print('Found image topic: {} with {} messages'.format(topic_name, num_img_msgs))
        if topic_name == flow_topic:
            num_img_msgs = topic_info.message_count
            print('Found flow topic: {} with {} messages'.format(topic_name, num_img_msgs))
    return num_event_msgs, num_img_msgs, num_flow_msgs


# Inspired by https://github.com/uzh-rpg/rpg_e2vid
def extract_rosbag(rosbag_path, output_path, event_topic, image_topic=None,
                start_time=None, end_time=None, zero_timestamps=False,
                is_color=False, sensor_size=None):
    topics = (event_topic, image_topic)
    event_msg_sum = 0
    num_msgs_between_logs = 25
    first_ts = -1
    t0 = -1
    
    if not os.path.exists(rosbag_path):
        print("{} does not exist!".format(rosbag_path))
        return
    with rosbag.Bag(rosbag_path, 'r') as bag:
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
        num_event_msgs, num_img_msgs, num_flow_msgs = get_rosbag_stats(bag, event_topic, image_topic)
        max_buffer_size = 1000000
        
        xs, ys, ts, ps = [], [], [], []
        max_buffer_size = 1000000
        last_ts, img_cnt = 0, 0
        
        f_timestamp = open(os.path.join(output_path, 'timestamps.txt'),'w')
        path_to_write_event_csv = os.path.join(output_path, 'events.csv')
        with open(path_to_write_event_csv, 'w', newline='') as f_events:
            for topic, msg, t in bag.read_messages():
                if first_ts == -1 and topic in topics:
                    timestamp = timestamp_float(msg.header.stamp)
                    first_ts = timestamp
                    if zero_timestamps:
                        timestamp = timestamp-first_ts
                    if start_time is None:
                        start_time = first_ts
                    start_time = start_time + first_ts
                    if end_time is not None:
                        end_time = end_time+start_time
                    t0 = timestamp
                    

                if topic == image_topic:
                    timestamp = timestamp_float(msg.header.stamp)-(first_ts if zero_timestamps else 0)
                    if is_color:
                        image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
                    else:
                        image = CvBridge().imgmsg_to_cv2(msg, "mono8")

                    
                    cv2.imwrite(os.path.join(output_path, 'frames', 'frame_{:010d}.png'.format(img_cnt)), image)
                    f_timestamp.write(str(img_cnt)+' '+str(timestamp)+'\n')
    
                    sensor_size = image.shape
                    img_cnt += 1
                    
                elif topic == event_topic:
                    event_msg_sum += 1
                    if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= num_event_msgs - 1:
                        print('Event messages: {} / {}'.format(event_msg_sum + 1, num_event_msgs))
                    
                    for e in msg.events:
                        timestamp = timestamp_float(e.ts)-(first_ts if zero_timestamps else 0)
                        xs.append(e.x)
                        ys.append(e.y)
                        ts.append(timestamp)
                        ps.append(1 if e.polarity else 0)

                        last_ts = timestamp
                        
                    writer = csv.writer(f_events, delimiter=' ')
                    for row in zip(ts, xs, ys, ps):
                        writer.writerow(row)
                    # writer.writerows(np.stack((ts, xs, ys, ps),1))
    
                    if (len(xs) > max_buffer_size and timestamp >= start_time) or (end_time is not None and timestamp >= start_time):
                        # print("Writing events")
                        if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
                            sensor_size = [max(xs), max(ys)]
                            print("Sensor size inferred from events as {}".format(sensor_size))     
 
                        del xs[:]
                        del ys[:]
                        del ts[:]
                        del ps[:]
                    if end_time is not None and timestamp >= start_time:
                        return
                    if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
                        sensor_size = [max(xs), max(ys)]
                        print("Sensor size inferred from events as {}".format(sensor_size))
                    
                    del xs[:]
                    del ys[:]
                    del ts[:]
                    del ps[:]
        print("Detect sensor size {}".format(sensor_size))
        f_timestamp.close()
        f_events.close()
        
        zf = zipfile.ZipFile(os.path.join(output_path, 'events.zip'), 'w', zipfile.zlib.DEFLATED)
        zf.write(path_to_write_event_csv, 'events.csv')
        zf.close()
        
        os.remove(path_to_write_event_csv)

def extract_rosbag_with_flow(rosbag_path, output_path, event_topic, image_topic=None,
                   flow_topic=None, start_time=None, end_time=None, zero_timestamps=False,
                is_color=False, sensor_size=None):

    topics = (event_topic, image_topic, flow_topic)
    event_msg_sum = 0
    num_msgs_between_logs = 25
    first_ts = -1
    t0 = -1
    if not os.path.exists(rosbag_path):
        print("{} does not exist!".format(rosbag_path))
        return
    with rosbag.Bag(rosbag_path, 'r') as bag:
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
        num_event_msgs, num_img_msgs, num_flow_msgs = get_rosbag_stats(bag, event_topic, image_topic, flow_topic)
        # Extract events to h5
        xs, ys, ts, ps = [], [], [], []
        max_buffer_size = 1000000
        # num_pos, num_neg, last_ts, 
        img_cnt, flow_cnt = 0, 0
        
        f_timestamp = open(os.path.join(output_path, 'timestamps.txt'),'w')
        path_to_write_event_csv = os.path.join(output_path, 'events.csv')
        with open(path_to_write_event_csv, 'w', newline='') as f_events:
            for topic, msg, t in bag.read_messages():
                if first_ts == -1 and topic in topics:
                    timestamp = timestamp_float(msg.header.stamp)
                    first_ts = timestamp
                    if zero_timestamps:
                        timestamp = timestamp-first_ts
                    if start_time is None:
                        start_time = first_ts
                    start_time = start_time + first_ts
                    if end_time is not None:
                        end_time = end_time+start_time
                    t0 = timestamp
                    

                if topic == image_topic:
                    timestamp = timestamp_float(msg.header.stamp)-(first_ts if zero_timestamps else 0)
                    if is_color:
                        image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
                    else:
                        image = CvBridge().imgmsg_to_cv2(msg, "mono8")

                    
                    cv2.imwrite(os.path.join(output_path, 'frames', 'frame_{:010d}.png'.format(img_cnt)), image)
                    f_timestamp.write(str(img_cnt)+' '+str(timestamp)+'\n')
    
                    sensor_size = image.shape
                    img_cnt += 1
                
                elif topic == flow_topic:
                    timestamp = timestamp_float(msg.header.stamp)-(first_ts if zero_timestamps else 0)

                    flow_x = np.array(msg.flow_x)
                    flow_y = np.array(msg.flow_y)
                    flow_x.shape = (msg.height, msg.width)
                    flow_y.shape = (msg.height, msg.width)
                    flow_image = np.stack((flow_x, flow_y), axis=0)

                    # ep.package_flow(flow_image, timestamp, flow_cnt)
                    np.savez_compressed(os.path.join(output_path, 'flow', 'flow_{:010d}.npz'.format(flow_cnt)), flow=flow_image)
         
                    flow_cnt += 1

                    
                elif topic == event_topic:
                    event_msg_sum += 1
                    if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= num_event_msgs - 1:
                        print('Event messages: {} / {}'.format(event_msg_sum + 1, num_event_msgs))
                    
                    for e in msg.events:
                        timestamp = timestamp_float(e.ts)-(first_ts if zero_timestamps else 0)
                        xs.append(e.x)
                        ys.append(e.y)
                        ts.append(timestamp)
                        ps.append(1 if e.polarity else 0)

                        last_ts = timestamp
                        
                    writer = csv.writer(f_events, delimiter=' ')
                    for row in zip(ts, xs, ys, ps):
                        writer.writerow(row)
                    # writer.writerows(np.stack((ts, xs, ys, ps),1))
    
                    if (len(xs) > max_buffer_size and timestamp >= start_time) or (end_time is not None and timestamp >= start_time):
                        # print("Writing events")
                        if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
                            sensor_size = [max(xs), max(ys)]
                            print("Sensor size inferred from events as {}".format(sensor_size))     
 
                        del xs[:]
                        del ys[:]
                        del ts[:]
                        del ps[:]
                    if end_time is not None and timestamp >= start_time:
                        return
                    if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
                        sensor_size = [max(xs), max(ys)]
                        print("Sensor size inferred from events as {}".format(sensor_size))
                    
                    del xs[:]
                    del ys[:]
                    del ts[:]
                    del ps[:]
        print("Detect sensor size {}".format(sensor_size))
        f_timestamp.close()
        f_events.close()
        
        zf = zipfile.ZipFile(os.path.join(output_path, 'events.zip'), 'w', zipfile.zlib.DEFLATED)
        zf.write(path_to_write_event_csv, 'events.csv')
        zf.close()
        
        os.remove(path_to_write_event_csv)
        
    


def extract_rosbags(rosbag_paths, output_dir, event_topic, image_topic, flow_topic=None,
        zero_timestamps=False, is_color=False, sensor_size=None):
    for path in rosbag_paths:
        bagname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, bagname)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        path_to_frames = os.path.join(out_path, 'frames')
        if not os.path.exists(path_to_frames):
            os.makedirs(path_to_frames)
            
        path_to_flows = os.path.join(out_path, 'flow')
        if not os.path.exists(path_to_flows) and flow_topic:
            os.makedirs(path_to_flows)
        print("Extracting {} to {}".format(path, out_path))
        
        if not flow_topic:
            extract_rosbag(path, out_path, event_topic, image_topic=image_topic,
                        zero_timestamps=zero_timestamps,
                        is_color=is_color, sensor_size=sensor_size)
        else:
            extract_rosbag_with_flow(path, out_path, event_topic, image_topic=image_topic,
                                flow_topic=flow_topic,
                                zero_timestamps=zero_timestamps,
                                is_color=is_color, sensor_size=sensor_size)

if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    
    cd utils
    python extract_rosbog.py
    extract and save frames and events for HQF dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="ROS bag file to extract or directory containing bags")
    parser.add_argument("--output_dir", default="/extracted_data", help="Folder where to extract the data")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--image_topic", default="/dvs/image_raw", help="Image topic (if left empty, no images will be collected)")
    parser.add_argument("--flow_topic", default="/dvs/flow_raw", help="Flow topic (if left empty, no images will be collected)")
    parser.add_argument('--zero_timestamps', action='store_true', help='If true, timestamps will be offset to start at 0')
    parser.add_argument('--is_color', action='store_true', help='Set flag to save frames from image_topic as 3-channel, bgr color images')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width',  type=int, default=None)
    args = parser.parse_args()

    args.path = '/rds/general/user/sl220/home/data/EVData/HQF/'
    args.output_dir = '/rds/general/user/sl220/home/data/EVData/HQF_extract/'
    args.zero_timestamps = True
    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        rosbag_paths = sorted(glob.glob(os.path.join(args.path, "*.bag")))
    else:
        rosbag_paths = [args.path]
    if args.height is None or args.width is None:
        sensor_size = None
    else:
        sensor_size = [args.height, args.width]

    extract_rosbags(rosbag_paths, args.output_dir, args.event_topic, args.image_topic, args.flow_topic,
             zero_timestamps=args.zero_timestamps, is_color=args.is_color,
            sensor_size=sensor_size)
    
    
    #{'/dvs/events': TopicTuple(msg_type='dvs_msgs/EventArray', message_count=2972, connections=1, frequency=32.56143837530665), '/dvs/image_raw': TopicTuple(msg_type='sensor_msgs/Image', message_count=2431, connections=1, frequency=22.749322695333582)}
    # no flow info in HQF data