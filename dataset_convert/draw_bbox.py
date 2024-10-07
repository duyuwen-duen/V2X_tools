import numpy as np
import cv2,os
import json
from pyquaternion import Quaternion
import argparse
from scipy.spatial.transform import Rotation as R
import math
import re

     
def project_to_image(points, K,l2c):
    """
    transform points (x,y,z) to image (h,w,1)
    """
    points = np.array(points).reshape((3, 1))
    R = l2c[:3,:3]
    T = l2c[:3,3]
    points_camera = np.dot(R, points) + T.reshape((3, 1))
    points_2d_homogeneous = np.dot(K, points_camera)
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d

def get_3dbbox_corners(label):
    location = [label['3d_location']['x'], label['3d_location']['y'], label['3d_location']['z']]
    size = [label['3d_dimensions']['l'],label['3d_dimensions']['w'],label['3d_dimensions']['h']]
    theta = label['rotation']
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    points = np.array([[-size[0]/2,-size[1]/2,-size[2]/2],
                    [size[0]/2,-size[1]/2,-size[2]/2],
                    [size[0]/2,size[1]/2,-size[2]/2],
                    [-size[0]/2,size[1]/2,-size[2]/2],
                    [-size[0]/2,-size[1]/2,size[2]/2],
                    [size[0]/2,-size[1]/2,size[2]/2],
                    [size[0]/2,size[1]/2,size[2]/2],
                    [-size[0]/2,size[1]/2,size[2]/2]])
    points = np.dot(Rz, points.T).T
    
    input_bbox = location + points
    
    return input_bbox


def draw_3dbbox_func(img, input_bbox,l2c,camera_intrinsic,name,videowriter):
    output_bbox = []
    for point in input_bbox:
        project_point = project_to_image(point, camera_intrinsic, l2c)
        output_bbox.append(project_point.flatten())
    output_bbox = np.array(output_bbox)
    points = output_bbox.astype(np.int32)
    # for i in range(0, 8):
    #     cv2.putText(img, str(i), tuple(points[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    is_inimage = (points[:,0] >= 0) & (points[:,0] < img.shape[1]) & (points[:,1] >= 0) & (points[:,1] < img.shape[0])
    if not np.any(is_inimage):
        return
    for i in range(0, 4):
        cv2.line(img, tuple(points[i]), tuple(points[(i+1)%4]), (225, 225, 0), 2)
        cv2.line(img, tuple(points[i+4]), tuple(points[(i+1)%4+4]), (225, 225, 0), 2)
        cv2.line(img, tuple(points[i]), tuple(points[i+4]), (225, 225, 0), 2)
    cv2.putText(img, name, tuple(points[7]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    videowriter.write(img)


def get_3dbbox_corners_in_image(input_bbox,l2c,camera_intrinsic):
    output_bbox = []
    for point in input_bbox:
        project_point = project_to_image(point, camera_intrinsic, l2c)
        output_bbox.append(project_point.flatten())
    output_bbox = np.array(output_bbox)
    points = output_bbox.astype(np.int32)
    return points

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='/home/ubuntu/duyuwen/ChatSim_v2x/data/standard_rcooper_mini', help='path of origin data')
    parse.add_argument('--road_seq', type=str, default='136-137-138-139')
    parse.add_argument('--road_name', type=str, default='136-0') 
    parse.add_argument('--data_type', type=str, default='train')
    args = parse.parse_args()
    data_dir = os.path.join(args.data_path,args.road_seq,args.road_name,args.data_type)  
    img_paths = os.path.join(data_dir,'image')
    label_paths = os.path.join(data_dir,'label/lidar_label')
    calib_dir = os.path.join(args.data_path,args.road_seq,args.road_name,'calib')
    
    with open(os.path.join(calib_dir,'camera_intrinsic', f'{args.road_name}.json'), 'r') as f:
        camera_info = json.load(f)
    camera_intrinsic = np.array(camera_info['intrinsic'])
    with open(os.path.join(calib_dir,'lidar2cam', f'{args.road_name}.json'), 'r') as f:
        l2c_data = json.load(f)
    l2c = np.array(l2c_data['lidar2cam']).reshape(4,4)

    output_paths = os.path.join(args.data_path.replace("/data/","/result/"),'3dbbox',args.road_seq,args.road_name,args.data_type)
        
    if not os.path.exists(output_paths):
        os.makedirs(output_paths)
    
    image_list = sorted(os.listdir(img_paths),key = lambda x: [int(num) if num else '' for num in re.findall(r'\d+', x)])
    label_list = sorted(os.listdir(label_paths),key = lambda x: [int(num) if num else '' for num in re.findall(r'\d+', x)])
    video_path = os.path.join(output_paths, 'video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(video_path,fourcc, 10.0, (1920,1200),True)
    for i in range(len(image_list)):
        img_path = os.path.join(img_paths, image_list[i])
        label_path = os.path.join(label_paths, label_list[i])
        output_path = os.path.join(output_paths, image_list[i])
        if not os.path.exists(label_path) or not os.path.exists(img_path):
            print(f"{label_path} or {img_path} not exist")
            continue
        labels = json.load(open(label_path))
        img = cv2.imread(img_path)
        for label in labels:
            name = label['type']
            input_bbox = get_3dbbox_corners(label)
            draw_3dbbox_func(img, input_bbox,l2c,camera_intrinsic,name,videowriter=videowriter)
            
                
        cv2.imwrite(output_path, img)
        print(f"save to {output_path}")
                
                
        
    
            