import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import Canvas
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import argparse,math,re
from label_tools import get_annos_all, save_annos,filter_out_img_box_center_new
from tqdm import tqdm


def copy_data(origin_path,dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for file in os.listdir(origin_path):
        origin_file = os.path.join(origin_path, file)
        name,ext = os.path.splitext(file)
        dst_file = os.path.join(dst_path, name.replace('.','_')+ext)
        os.system('cp ' + origin_file + ' ' + dst_file)

def generate_image_label(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,save_path):
    road_name, cam_id = cam_name.rsplit('-', 1)
    cam_id = int(cam_id) 
    for seq_name in tqdm(sorted(seq_list)):
        origin_image_path = os.path.join(road_seq_path, road_name, seq_name,f'cam-{cam_name.split("-")[-1]}')
        road_seq_label_path = re.sub(r'(?<!/data/)/data/([^/]+)$', '/label/\\1', road_seq_path)
        origin_label_path = os.path.join(road_seq_label_path, road_name, seq_name)
        if int(seq_name.split('-')[-1]) in val_data_seq:
            dst_image_path = os.path.join(save_path,road_seq,cam_name, 'val', 'image')
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'val', 'label/lidar_label')
        elif int(seq_name.split('-')[-1]) in test_data_seq:
            dst_image_path = os.path.join(save_path,road_seq,cam_name, 'test', 'image')
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'test', 'label/lidar_label')
        else:
            dst_image_path = os.path.join(save_path,road_seq,cam_name, 'train', 'image')
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'train', 'label/lidar_label')
        copy_data(origin_image_path, dst_image_path)
        copy_data(origin_label_path, dst_label_path) 
       

def get_calib(calib_path, road_name, cam_id):
    with open(os.path.join(calib_path, 'lidar2cam',f'{road_name}.json'), 'r') as f:
        data = json.load(f)
    camera_intrinsic = data[f'cam_{cam_id}']['intrinsic']
    P = camera_intrinsic
    lidar2cam = data[f'cam_{cam_id}']['extrinsic']
    t_velo2cam = np.array([[lidar2cam[0][3]], [lidar2cam[1][3]], [lidar2cam[2][3]],])
    r_velo2cam = np.array([lidar2cam[0][:3], lidar2cam[1][:3], lidar2cam[2][:3]])
    return P,r_velo2cam, t_velo2cam

def label_lidar2camera(lidar_label_path, P, r_velo2cam, t_velo2cam):
    gt_names, gt_boxes, other_infos = get_annos_all(lidar_label_path)
    #type:list, list
    new_gt_names,new_gt_bboxes, new_other_infos = filter_out_img_box_center_new(gt_names,gt_boxes, other_infos, P, r_velo2cam, t_velo2cam)
    new_gt_names = list(new_gt_names)
    new_label = save_annos(new_gt_names, new_gt_bboxes, new_other_infos)
    return new_label

def label_filter(origin_label_path, dst_label_path, P, r_velo2cam, t_velo2cam):
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)
    for file in os.listdir(origin_label_path):
        origin_file = os.path.join(origin_label_path, file)
        processed_label = label_lidar2camera(origin_file, P, r_velo2cam, t_velo2cam)
        name,ext = os.path.splitext(file)
        dst_file = os.path.join(dst_label_path, name.replace('.','_')+ext)
        with open(dst_file, 'w') as f:
            json.dump(processed_label, f, indent=4)

def generate_filtered_label(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,save_path):
    road_name, cam_id = cam_name.rsplit('-', 1)
    cam_id = int(cam_id) 
    for seq_name in sorted(seq_list):
        road_seq_label_path = re.sub(r'(?<!/data/)/data/([^/]+)$', '/label/\\1', road_seq_path)
        origin_label_path = os.path.join(road_seq_label_path, road_name, seq_name)
        if int(seq_name.split('-')[-1]) in val_data_seq:
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'val', 'label/camera_label')
        elif int(seq_name.split('-')[-1]) in test_data_seq:
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'test', 'label/camera_label')
        else:
            dst_label_path = os.path.join(save_path,road_seq,cam_name, 'train', 'label/camera_label')
        P, r_velo2cam, t_velo2cam = get_calib(calib_path, road_name, cam_id)
        label_filter(origin_label_path, dst_label_path, P, r_velo2cam, t_velo2cam) 

def generate_calib(calib_path, road_name, cam_id,save_calib_path):
    
    if not os.path.exists(os.path.join(save_calib_path,'camera_intrinsic')):
        os.makedirs(os.path.join(save_calib_path,'camera_intrinsic'))
    if not os.path.exists(os.path.join(save_calib_path,'lidar2cam')):
        os.makedirs(os.path.join(save_calib_path,'lidar2cam'))
    if not os.path.exists(os.path.join(save_calib_path,'lidar2world')):
        os.makedirs(os.path.join(save_calib_path,'lidar2world'))
        
    with open(os.path.join(calib_path, 'lidar2cam',f'{road_name}.json'), 'r') as f:
        data = json.load(f)
    camera_intrinsic = data[f'cam_{cam_id}']['intrinsic']
    lidar2cam = data[f'cam_{cam_id}']['extrinsic']
    if 'distortion' in data[f'cam_{cam_id}']:
        distortion = data[f'cam_{cam_id}']['distortion']
    else:
        distortion = []
        
    with open(os.path.join(calib_path, 'lidar2world',f'{road_name}.json'), 'r') as f:
        data = json.load(f)
    lidar2world = np.eye(4)
    lidar2world[:3,:3] = np.array(data['rotation']).reshape(3,3)
    lidar2world[:3,3] = np.array(data['translation'])
    
    camera_info = {
        'intrinsic': camera_intrinsic,
        'distortion': distortion
    }
    lidar2cam_info = {
        'lidar2cam': lidar2cam
    }
    lidar2world_info = {
        'lidar2world': lidar2world.tolist()
    }
    with open(os.path.join(save_calib_path,'camera_intrinsic',f'{road_name}-{cam_id}.json'), 'w') as f:
        json.dump(camera_info, f)
    with open(os.path.join(save_calib_path,'lidar2cam',f'{road_name}-{cam_id}.json'), 'w') as f:
        json.dump(lidar2cam_info, f)
    with open(os.path.join(save_calib_path,'lidar2world',f'{road_name}-{cam_id}.json'), 'w') as f:
        json.dump(lidar2world_info, f)


def generate_coop_label(seq_list,road_seq_path, road_name,save_coop_path,val_data_seq, test_data_seq):
    for seq_name in sorted(seq_list):
        road_seq_path = re.sub(r'(?<!/data/)/data/([^/]+)$', '/label/\\1', road_seq_path)
        origin_label_path = os.path.join(road_seq_path, road_name, seq_name)
        if int(seq_name.split('-')[-1]) in val_data_seq:
            dst_label_path = os.path.join(save_coop_path, 'val', 'label/lidar_label')
        elif int(seq_name.split('-')[-1]) in test_data_seq:
            dst_label_path = os.path.join(save_coop_path, 'test', 'label/lidar_label')
        else:
            dst_label_path = os.path.join(save_coop_path, 'train', 'label/lidar_label')
            
        copy_data(origin_label_path, dst_label_path)
    
    
def generate_lidar(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,save_path):  
    road_name, cam_id = cam_name.rsplit('-', 1)
    for seq_name in sorted(seq_list):
        origin_lidar_path = os.path.join(road_seq_path.replace('label','data'), road_name, seq_name,'lidar')
        if int(seq_name.split('-')[-1]) in val_data_seq:
            dst_lidar_path = os.path.join(save_path,road_seq,cam_name, 'val', 'lidar')
        elif int(seq_name.split('-')[-1]) in test_data_seq:
            dst_lidar_path = os.path.join(save_path,road_seq,cam_name, 'test', 'lidar')
        else:
            dst_lidar_path = os.path.join(save_path,road_seq,cam_name, 'train', 'lidar')
        copy_data(origin_lidar_path, dst_lidar_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TUM dataset to RCooper dataset')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/duyuwen/ChatSim_v2x/data/rcooper_tum', help='path to the dataset directory')
    parser.add_argument('--save_path', type=str, default='/home/ubuntu/duyuwen/ChatSim_v2x/result/standard_tum', help='path to save the converted dataset')
    args = parser.parse_args()
    
    data_name = ['train', 'val', 'test']
    split_path = os.path.join(args.data_path, 'split_data.json')
    calib_path = os.path.join(args.data_path, 'calib')
    image_path = os.path.join(args.data_path, 'data')
    label_path = os.path.join(args.data_path, 'label')
    road_seq_list = os.listdir(image_path)
    print(f'road_seq_list: {road_seq_list}')
    
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    for road_seq in sorted(road_seq_list):
        print(f'=================== Processing road sequence {road_seq} ===================')
        val_data_seq = split_data[road_seq]['val']
        test_data_seq = split_data[road_seq]['test']
        road_seq_path = os.path.join(image_path, road_seq)
        road_name_list = os.listdir(road_seq_path)
        # decide the camera number
        seq_list = os.listdir(os.path.join(road_seq_path, road_name_list[0]))
        if "sim" in road_seq_path: 
            cam_num = len(os.listdir(os.path.join(road_seq_path, road_name_list[0], seq_list[0])))
        else: # not include lidar directory
            cam_num = len(os.listdir(os.path.join(road_seq_path, road_name_list[0], seq_list[0]))) - 1
        #road_name_list: 105,106,115.etc
        for road_name in sorted(road_name_list):
            print(f'################## Processing road name {road_name} #######################')
            for cam_id in range(cam_num):
                print(f'============== Processing {cam_id} camera ==============')
                cam_name = road_name + '-' + str(cam_id)

                print(f'============== Processing image and label ==============')
                generate_image_label(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,args.save_path)

                print(f'============== Processing filtered label ==============')
                generate_filtered_label(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,args.save_path)

                print(f'============== Processing calib ==============')
                # generate calib for each camera
                generate_calib(calib_path, road_name, cam_id,os.path.join(args.save_path,road_seq,cam_name, 'calib'))

                if 'sim' in road_seq_path:
                    continue
                print(f'============== Processing lidar ==============')
                generate_lidar(seq_list, road_seq_path, road_seq, cam_name, val_data_seq, test_data_seq,args.save_path)
        
        # generate coop label for the road sequence
        print(f'============== Processing coop label ==============')
        generate_coop_label(seq_list,road_seq_path, 'coop', os.path.join(args.save_path,road_seq,'coop'), val_data_seq, test_data_seq)
                
                    