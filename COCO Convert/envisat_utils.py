# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:35:30 2020
To do: Add timing and debug messages
        For data in txt for which a jpg doesnot exist. A problem! See what to do! Change index based implmentation if necessary, or just remove lines from the data
        to change the implementaiton, either post filter and delete or add index map to improve gt_data writing code

        Filename check added- two sided check:
            1. Check if filename exists and only then add idx in the map
            2. Access the list using the index map only. checks if filename exists in index map
@author: Kuldeep Barad
"""
import os
import numpy as np
import cv2
import re
from matplotlib import pyplot as plt
import csv
import json
import glob
import time
import shutil


class Camera:

    def __init__(self,img_size):
        #size : 2-Tuple (X,Y) (e.g. (1920,1200) )
        self.nu = img_size[0] # number of horizontal[pixels]
        self.nv = img_size[1] # number of vertical[pixels]
        self.fx = 0.007736896413/2  # focal length[m]
        self.fy = 0.007736896413/2  # focal length[m]
        ppx = 0.0000055  # horizontal pixel pitch[m / pixel]
        ppy = ppx  # vertical pixel pitch[m / pixel]
        self.fpx = self.fx / ppx  # horizontal focal length[pixels]
        self.fpy = self.fy / ppy  # vertical focal length[pixels]
        k = [[self.fpx,     0,      self.nu / 2],
             [0,        self.fpy,   self.nv / 2],
             [0,            0,          1       ]]
        self.K = np.array(k)
    """" Utility class for accessing camera parameters. """


class ENVISAT_DS:
    def __init__(self,train_path,val_path,test_path, camera_obj, **kwargs):
        ''' Assumed: each folder contains images and data files'''

        self.dirs = {'train':train_path,
                     'test':test_path,
                     'val': val_path,
                     'root': os.path.abspath(os.path.join(train_path,'..'))
                     }

        self.time =  time.time()
        self.datasets = self.get_dataset_list(selection ='test')
        self.Camera = camera_obj

        self.n_subfolder = kwargs['n_subfolder'] #if images divided into subfolders, n = no. images per subfolder
        images_in_subfolders = True if self.n_subfolder else False
        self.filenames = self.get_all_img_filenames(images_in_subfolders,self.n_subfolder)

        dt = time.time() -self.time
        print(f"Initial loading :{dt}")

        #Load model points
        self.model3D = self.load_3Dmodel(kwargs['model']['path'], data_unit = kwargs['model']['unit'])

        self.n_samples={}
        self.filename_header = kwargs['imgname_header']

        '''Note this can be directly written as bbox json for SSDLite'''
        dt = time.time() -self.time
        print(f"Initial gt data loading :{dt}")
        self.gt_data,self.index_maps = self.get_gt_data(self.datasets, **kwargs)

        self.results_loaded = False




    def get_all_img_filenames(self, subfolders=False, n_subfolder = None):
        f_dicts = {}
        for img_set in self.datasets:
            if subfolders:
                paths = glob.glob(os.path.join(self.dirs[img_set],'**','*.jp*g'))
            else:
                paths = glob.glob(os.path.join(self.dirs[img_set],'*.jp*g'))
            f_dicts[img_set] = [path.split('\\')[-1] for path in paths]
        dt = time.time() -self.time
        print(f"Filename loading :{dt}")
        return f_dicts

    def check_extension(self, filename, add = False, remove = False):
        split_ext =  filename.split('.')
        ext = ''
        if len(split_ext)>1:
            ext = split_ext[-1]
        if add:
            ext = ext if ext!='' else 'jpg'
            return f"{filename}.{ext}", True
        elif remove:
            return split_ext[0], False
        else:
            exists = False if ext =='' else True
            return filename, exists

    def get_name_from_id(self,id):
        return f"{self.imgname_header}{id}.jpg"

    def get_id_from_name(self, name):
        split_ext =  name.split('.')
        img_id = int(split_ext[0].split(self.filename_header)[-1])
        return img_id

    def get_name_from_path(self, filepath):
        path_type1 = filepath.split('/')
        path_type2 = filepath.split('\\')
        if len(path_type1)>1:
            return path_type1[-1]
        else:
            return path_type2[-1]

    def get_image(self,image_set, flip=True, img_name=None, idx= None):
        '''
            image_set = 'train', 'val' or 'test'

        '''
        img_exists=False
        if img_name ==None:
            if idx != None:
                img_name= self.get_name_from_id(idx)
            else:
                Exception('no info provided')
        else:
            idx = self.get_id_from_name(img_name)
        img_name,_ = self.check_extension(img_name, add= True)

        if img_name in self.filenames[image_set]:
            img_exists = True
            file_dir = self.get_filepath(image_set,idx)
            file_path = os.path.join(file_dir,img_name)
            img = cv2.imread(file_path)
            if flip:
                #Images are stored in BGR and not RGB so need to flip
                rgb_img = np.fliplr(img.reshape(-1,3)).reshape(img.shape)
            else:
                rgb_img = img
            return rgb_img, img_exists
        else:
            print(f"Image {img_name} not found")
            return None,img_exists


    def get_img_gt(self, image_set, img_name=None, idx = None):
        '''
            image_set = 'train', 'val' or 'test'

        '''
        if idx == None:
            idx = img_name.split(self.filename_header)[1]
        return self.gt_data[image_set][idx]


    def get_dataset_list(self,selection):
        '''selection = train, val, test, all
            ds = referred tp dataset
        '''
        ds_list = []
        if selection == 'train' or selection == 'all':
            ds_list.append('train')

        if selection == 'val' or selection == 'all':
            ds_list.append('val')

        if selection == 'test' or selection == 'all':
            ds_list.append('test')

        return ds_list

    def load_3Dmodel(self, filepath, format='csv',data_unit=None):
        '''
        filepath = 3d model features file, currently on csv supported.
        data_unit= 'm', 'cm','mm'
        default unit for returned array is m. Multplier is defined by unit specified in kwargs
        returns a numpy array of size (3,n_features)
        '''
       # To do: JSON support

        data_unit.lower()
        if data_unit =='m':
            multiplier=1
        elif data_unit == 'cm':
            multiplier = 0.01
        elif data_unit == 'mm':
            multiplier = 0.001
        else:
            Exception('Unit for 3D model points not understsood. Please add value for unit argument in the function call from \n \'m\',\'cm\',\'mm\' ')

        if format == 'csv':
            data = []
            with open(filepath,'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(row)

            x_3D = np.array(data,dtype=np.float64)[1:,:]*multiplier
        else:
            Exception('format not supported yet')
            data = None

        return x_3D


    ## Deprecate this when moving from Envisat_set1 dataset and make changes to dataformat implementation in parsing txt!!!
    def get_pose_from_original_data(self,data_file,image_set):
        file_obj = open(data_file,'r')
        txt_data = file_obj.readlines()
        for idx, img_data in enumerate(txt_data):
            filename = self.filename_header+str(idx)+'.jpg' #Expected filename
            if filename in self.filenames[image_set]:
                index_map = self.index_maps[image_set]
                mapped_index = index_map[filename] #Ex
                pose = [float(num) for num in (img_data.split(' ')[1:5])]
                t_BC = [0,0,pose[0]]
                corrected_euler = self.apply_Euler_correction(pose[1:4])
                q_BC = list(Kinematics.eul2quat(np.radians(corrected_euler)))
                self.gt_data[image_set][mapped_index]['pose'] = {'r': t_BC, 'q':q_BC}
            else:
                print(f"{filename} does not exist")

        dt = time.time() - self.time
        print(f"pose processed {dt}")


    def apply_Euler_correction(self, Euler_angles):
        ''' Euler angles: [EA1,EA2,EA3] - Cinema 4D HPB to RPY '''
        if 180 + Euler_angles[2]<=180:
            corrected = [Euler_angles[0], -Euler_angles[1], 180 + Euler_angles[2]]
        else:
            corrected = [Euler_angles[0], -Euler_angles[1], Euler_angles[2]-180]

        return corrected


    def get_gt_data(self,ds_list,**kwargs):
        '''


        Parameters
        ----------
        ds_list : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        data_dict : TYPE
            DESCRIPTION.

        '''
        ''' load txt files '''
        data_dict={} # set wise data dict
        index_maps= {}
        for ds in ds_list:
            data_file = os.path.join(self.dirs[ds],kwargs['datafile'][ds])
            file_obj = open(data_file,'r')
            txt_data = file_obj.readlines() #Returns a list of str, ideally per image
            file_obj.close()
            data_dict[ds], index_maps[ds]  = self.parse_txt_data(txt_data,self.filenames[ds], **kwargs)
            dt = time.time() -self.time
            print(f"gt data Iteration :{dt}")
            self.n_samples[ds] = len(data_dict[ds])
        return data_dict, index_maps


    def parse_txt_data(self,data_list, filenames, **kwargs):
        '''


        Parameters
        ----------
        data_list : TYPE
            DESCRIPTION.
        filenames: Filenames for all images of the set that is passed in the data_list
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        img_dicts : TYPE
            DESCRIPTION.

        '''
        img_dicts = []
        index_map = {}  #Dict mapping filenames to list indeices

        for idx,data in enumerate(data_list):
            img_data = data.split(' ')

            data_format = kwargs['data_format']

            ''' Use kwargs['data_format'] to parse'''
            fields_data= {}

            #Get list of indices where list/number representing each field is located
            for key in list(data_format):
                field_location = self.infer_field_locations(data_format[key])
                fields_data[key] = img_data[field_location] if field_location != None else []

            filename = self.filename_header+str(idx)+'.jpg'
            feature_dicts, bbox = self.get_kp_bbox(fields_data['features'])
            '''
            CHANGE THIS FOR POSE FIELD WHEN NEEDED
            '''
            attitude_data = [float(num) for num in fields_data['attitude']]
            position_data = [float(num) for num in fields_data['z_boresight']]

            if len(attitude_data) ==3:
                euler_corrected = np.radians(self.apply_Euler_correction(attitude_data))
                attitude_data = list(Kinematics.eul2quat(np.array(euler_corrected)))
            this_img_dict = {
                'id':idx if fields_data['id']!=None else fields_data['id'],           # CHANGE IF IDs are not sequential or starting from 0!!!!!
                'filename': f"{filename}",
                'features': feature_dicts,
                'bbox': bbox,
                'pose': {
                    'r':[0,0,position_data[0]],
                    'q':attitude_data
                    },
                'illumination': fields_data['AzEl']
                }
            img_dicts.append(this_img_dict)
            if filename in filenames:
                index_map[filename] = idx

        return img_dicts, index_map

    def infer_field_locations(self, data_format):
        '''


        Parameters
        ----------
        data_format : TYPE
            DESCRIPTION.

        Returns
        -------
        field_loc : TYPE
            DESCRIPTION.

        '''
        if data_format!= None:
            start_index = data_format[0]
            len_fields = data_format[1]
            field_loc = slice(start_index, (start_index+len_fields)) if len_fields!= 0 else start_index
            return field_loc
        else:
            return None

    def get_kp_bbox(self, kp_list, id_included = False):
        '''


        Parameters
        ----------
        kp_list : TYPE
            DESCRIPTION.
        id_included : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        features : TYPE
            DESCRIPTION.
        bbox : TYPE
            DESCRIPTION.

        '''
        # Rigid implementation for data in format [id,x,y,id,x,y] or [x,y,x,y]

        n = int(len(kp_list)/2) if not id_included else int(len(kp_list)/3)
        u_list = []
        v_list = []
        features = []
        for i in range(n):
            u = float(kp_list[2*i])
            v = float(kp_list[2*i+1])
            u_list.append(u)
            v_list.append(v)
            coords = [u,v]
            visibility = 0 if u>self.Camera.nu or u<0 or v>self.Camera.nv or v<0 else 1
            features.append({ 'ID' : i, 'Coordinates': coords, 'Visibility':visibility})

        uv_pts= [u_list,v_list]
        bbox = self.compute_bbox(uv_pts)
        return features,bbox

    def compute_bbox(self, uv_pt, margin = 5):
        '''


        Parameters
        ----------
        uv_pt : TYPE
            DESCRIPTION.
        margin : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        list
            DESCRIPTION.

        '''
        #margin:percent : default  5% relaxed in w/h
        relax_margin_u= (margin/100)*(max(uv_pt[0])-min(uv_pt[0]))
        relax_margin_v= (margin/100)*(max(uv_pt[1])-min(uv_pt[1]))
        u_max = max(uv_pt[0])+relax_margin_u if max(uv_pt[0])+relax_margin_u <=self.Camera.nu else self.Camera.nu
        u_min = min(uv_pt[0])-relax_margin_u if min(uv_pt[0])-relax_margin_u >= 0 else 0
        v_max = max(uv_pt[1])+relax_margin_v if (max(uv_pt[1])+relax_margin_v <=self.Camera.nv) else self.Camera.nv
        v_min = min(uv_pt[1])-relax_margin_v if min(uv_pt[1])-relax_margin_v >= 0 else 0

        return [u_min, u_max, v_min, v_max]


    def visualize (self,image_set, img_name=None, idx=None, kps=True, bbox=True, skeleton=True, ax = None ):
        '''


        Parameters
        ----------
        image_set : TYPE
            DESCRIPTION.
        img_name : TYPE, optional
            DESCRIPTION. The default is None.
        idx : TYPE, optional
            DESCRIPTION. The default is None.
        kps : TYPE, optional
            DESCRIPTION. The default is True.
        bbox : TYPE, optional
            DESCRIPTION. The default is True.
        skeleton : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        '''
        if idx!= None:
            gt_dict = self.get_img_gt(image_set,idx=idx)
            img_name = self.filename_header+str(idx)
        else:
            gt_dict = self.get_img_gt(img_name=img_name)

        uv_pt = [[feature['Coordinates'][0] for feature in gt_dict['features']],
                 [feature['Coordinates'][1] for feature in gt_dict['features']]]


        img,exists = self.get_image(image_set,img_name=img_name)
        if exists:
            if ax == None:
                fig = plt.figure()
                ax =fig.gca()
            ax.imshow(img)
            if bbox:
                plot_bbox(gt_dict['bbox'],ax)

            ax.scatter(uv_pt[0],uv_pt[1],c = 'b', s=3)

            for k_feat in range(len(uv_pt[0])):
                ax.text(uv_pt[0][k_feat],uv_pt[1][k_feat],(' '+str(k_feat+1)), fontsize =7, c = 'y')

            plt.show()

    def check_projection_gt(self,image_set, img_name=None, idx=None):
        '''
        To check gt pose projection on image

        Parameters
        ----------
        image_set : TYPE
            DESCRIPTION.
        img_name : TYPE, optional
            DESCRIPTION. The default is None.
        idx : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if idx!= None:
            gt_dict = self.get_img_gt(image_set,idx=idx)
            img_name = self.filename_header+str(idx)
        else:
            gt_dict = self.get_img_gt(img_name=img_name)

        gt_uv_pt = [[feature['Coordinates'][0] for feature in gt_dict['features']],
                 [feature['Coordinates'][1] for feature in gt_dict['features']]]

        x_3d = self.model3D
        uv_pt= Projection.project_to_image(gt_dict['pose']['q'],gt_dict['pose']['r'], Camera = self.Camera, r_B = x_3d)

        img,exists = self.get_image(image_set,img_name=img_name)
        if exists:
            fig = plt.figure()
            ax =fig.gca()
            ax.imshow(img)
            ax.scatter(uv_pt[0],uv_pt[1],c = 'r', s=15)
            ax.scatter(gt_uv_pt[0],gt_uv_pt[1],marker='x',c = 'y', s=9, alpha =0.9)
            for k_feat in range(x_3d.shape[1]):
                plt.text(uv_pt[0][k_feat],uv_pt[1][k_feat],(' '+str(k_feat+1)), fontsize =7, c = 'y')

    def write2JSON(self, image_set, n_max, save_dir = None, filename=None):
        '''
        Saves json file(s) with name=filename in root/json_data/bbox/
        Parameters
        ----------
        image_set : 'train', 'val', 'test', 'all'
           'all' = [ 'train','test','val']
        filename: when set to 'default', saves file with name {image_set}.json e.g. train.json
        n_max: Maximum number of examples in a set
        '''
        valid_sets = {'train':'train_bbox.json','val':'val_bbox.json','test':'test_bbox.json'}
        if image_set in list(valid_sets):
            set_list = [image_set]
        elif image_set == 'all':
            set_list = list(valid_sets)
        else:
            Exception('Entered value for image set not valid')
            return

        #Copy

        if save_dir == None:
            save_dir = os.path.join(self.dirs['root'],'json_data','bbox')

        if filename!=None and filename.split('.')[-1] !='json':
            filename +='.json'

        for img_set in set_list:
            json_name = filename if filename != None else valid_sets[img_set]
            filepath = os.path.join(save_dir,json_name)
            json_data= []
            n=0
            for gt in self.gt_data[img_set]:
                if gt['filename'] in self.filenames[img_set]:
                    json_data.append(gt)
                    n+=1
            with open(filepath, 'w') as jsonfile:
                json.dump(json_data,jsonfile)

        return

    def get_filepath(self, img_set, idx):
        n_subfolder=self.n_subfolder
        if n_subfolder == None:
            return os.path.join(self.dirs[img_set])
        else:
            subfolder = DataFiles.get_subfolder(idx,n_subfolder)
            return os.path.join(self.dirs[img_set],subfolder)


    def load_HRNet_results(self, filepath):
        with open(filepath, 'r') as jsonfile:
            results = json.load(jsonfile)
            self.results_loaded=True
        return results

    def get_result_kps(self, result):
        kps = result['keypoints']
        x = []
        y= []
        scores = []
        num_kps = int(len(kps)/3)
        for idx in range(0,num_kps):
            x.append(kps[3*idx])
            y.append(kps[3*idx+1])
            scores.append(kps[3*idx+2])
        return [x,y], scores

    def visualize_result(self, results, img_set, input_size=(256,256), img_name= None, img_id = None,**kwargs):
        if img_id==None:
            if img_name==None:
                Exception('img_name or img_id must be provided')
            img_id = (img_name.split('.')[0]).split('frame')[-1] if img_name.split('.')[-1]=='.jpg' else img_name.split('frame')[-1]
        res_idmap = self._get_index_map(results,'image_id')
        result = results[res_idmap[img_id]]

        #GT
        gt_dict = self.get_img_gt(img_set,idx=img_id)
        gt_kps = [[feature['Coordinates'][0] for feature in gt_dict['features']],
                 [feature['Coordinates'][1] for feature in gt_dict['features']]]

        visibility = [feature['Visibility'] for feature in gt_dict['features']]

        # Predictions
        pred_kps, scores = self.get_result_kps(result)

        #visualize
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(pred_kps[0],pred_kps[1],c='r',marker='x',s =4)
        self.visualize(img_set,idx=img_id, ax= ax, bbox=False)
        gt_kps.append(visibility)
        pred_kps.append(scores)
        #Calculate Errors
        errors, avg_err,norm_error = Projection.calc_error(gt_kps,pred_kps,result['scale'])
        print(f" average error = {avg_err} \n normalized error = {norm_error}")




    def _get_index_map(self,dicts_list,key):
        '''
        return index map dict relating position in list and a value 'key
        of each dict in dict_lists

        '''
        id_map = {}
        for idx,entry in enumerate(dicts_list):
            id_map[entry[key]] = idx

        return id_map



class Projection():
    def project_to_image(q,r,Camera,r_B = np.ndarray([])):
        '''
            q = quaternion attitude [ q0, q1, q2, q3] = q0 + q_vec
            r = relative translation vector in camera frame (R_BC)
            r_B = Body frame points (Usually model points supplied for projection. By default, if no model points are provided the principle axes are projected)
        '''
        # default projection of reference axes unless feature points are provided in r_B
        if (r_B.size ==0):
    #    Reference axes in body frame homogeneous if no moidel points are provided
            p_axes = np.array([[0, 0, 0, 1],
                               [1, 0, 0, 1],
                               [0, 1, 0, 1],
                               [0, 0, 1, 1]])
    #        Reference points in body frame
            r_B_pts = np.transpose(p_axes)
        else:
            r_B_pts = np.expand_dims((r_B,1)) if r_B.size==1 else np.vstack((r_B,np.ones(r_B.shape[1])))

        R_BC = np.transpose(Kinematics.quat2dcm(q))
        t_C  = np.expand_dims(r,1)


        Pose_mat = np.hstack((R_BC,t_C))
        uvw= Camera.K.dot(np.dot(Pose_mat, r_B_pts))
        uv_img = uvw/uvw[2]

        return uv_img

    def calc_error(gt, pred,scale):
        # Return  [x_gt,x_pred,y_gt,y_pred,dx,dy,dr,score]
        n = len(gt[0])
        data_w_err = {}
        total_dr = 0
        total_norm_dr = 0

        for i in range(0,n):
            x_gt = gt[0][i]
            y_gt = gt[1][i]
            visibility = gt[2][i]
            x_pred = pred[0][i]
            y_pred = pred[1][i]
            score = pred[2][i]

            dx = abs(x_pred-x_gt)
            dy = abs(y_pred - y_gt)
            dr = float(np.sqrt(dx**2+dy**2))

            dx_normalized = dx/scale[0]
            dy_normalized = dy/scale[0]
            dr_normalized = float(np.sqrt(dx_normalized**2 + dy_normalized**2))

            total_dr += dr
            total_norm_dr += dr_normalized

            if visibility ==0:
                data_w_err[str(i)]= list(np.zeros((11,)))
            else:
                data_w_err[str(i)] = [x_gt,x_pred,y_gt,y_pred,score,dx,dy,dr,dx_normalized, dy_normalized, dr_normalized]

        avg_err = total_dr/n
        avg_norm_err=  total_norm_dr/n

        return data_w_err, avg_err, avg_norm_err

class Kinematics:
    def eul2quat(eul,seq='ZYX'):
        ''' Implementation of eul2quat in robotics toolbar MATLAB
            eul = np array of 3-euler angles in radians
            seq = XYZ, ZYX, ZYZ or lower cases of the same
            Verified with MATLAB

        '''
        # Check Valid Rotation Sequence
        valid_seq = ['XYZ','ZYX','ZYZ']
        if not seq.isupper():
            seq.upper()

        # to support multiple EA sequences, generalize single sequence to an array of an extra dim
        # Preallocate q zeros and adjust eul input
        if len(eul.shape)==1:
            eul = np.expand_dims(eul,0)
            q = np.zeros((1,4),dtype=type(eul[0]))
        elif len(eul.shape)==2:
            q = np.zeros((eul.shape[1],4),dtype=type(eul[0]))
        else:
            Exception('Cannot handle the array provided. Make sure the array is max two dimensional')
            return


        # Compute sines and cosines of half angles
        c = np.cos(eul/2)
        s = np.sin(eul/2)

        # The parsed sequence will be in all upper-case letters and validated
        if seq in valid_seq:
            if seq =='ZYX':
                q =np.array( [c[:,0]*c[:,1]*c[:,2]+s[:,0]*s[:,1]*s[:,2],
                    c[:,0]*c[:,1]*s[:,2]-s[:,0]*s[:,1]*c[:,2],
                    c[:,0]*s[:,1]*c[:,2]+s[:,0]*c[:,1]*s[:,2],
                    s[:,0]*c[:,1]*c[:,2]-c[:,0]*s[:,1]*s[:,2]])

            if seq =='ZYZ':
                q = np.array([c[:,0]*c[:,1]*c[:,2]-s[:,0]*c[:,1]*s[:,2],
                    c[:,0]*s[:,1]*s[:,2]-s[:,0]*s[:,1]*c[:,2],
                    c[:,0]*s[:,1]*c[:,2]+s[:,0]*s[:,1]*s[:,2],
                    s[:,0]*c[:,1]*c[:,2]+c[:,0]*c[:,1]*s[:,2]])

            if seq =='XYZ':
                q = np.array([c[:,0]*c[:,1]*c[:,2] - s[:,0]*s[:,1]*s[:,2],
                    s[:,0]*c[:,1]*c[:,2] + c[:,0]*s[:,1]*s[:,2],
                    -s[:,0]*c[:,1]*s[:,2] + c[:,0]*s[:,1]*c[:,2],
                    c[:,0]*c[:,1]*s[:,2] + s[:,0]*s[:,1]*c[:,2]])

            q = np.transpose(q)

            if q.shape[0]==1:
                return q[0]
            else:
                return q
        else:
            Exception('Valid Euler angle sequence was not provided')

    def quat2dcm(q):

        """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
        #Verified with MATLAB
        # normalizing quaternion
        q = q/np.linalg.norm(q)

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        dcm = np.zeros((3, 3))

        dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
        dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
        dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

        dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
        dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

        dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
        dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

        dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
        dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

        return dcm


class DataFiles:
    def create_subfolders(img_dir, n_files, n_subfolder = 1000):
        n = int(np.ceil(n_files/n_subfolder))
        for i in range(0,n):
            folder_name = str((i*n_subfolder))+'-'+str((i+1)*n_subfolder-1)
            fpath = os.path.join(img_dir,folder_name)
            if not os.path.exists(fpath):
                os.mkdir(fpath)

    def move_file(idx, file_path, subfolders_dir, n_subfolder=1000):
        '''
        file convention:
            frame{idx}.jpg
        '''
        subfolder = DataFiles.get_subfolder(idx,n_subfolder)
        target_path = os.path.join(subfolders_dir,subfolder)
        if os.path.exists(file_path):
            shutil.move(file_path,target_path)
        return

    def get_subfolder(idx, n_subfolder):
        idx_range = (int(np.floor(idx/n_subfolder))*n_subfolder,int(np.ceil(idx/n_subfolder))*n_subfolder)
        if idx_range[0] == idx_range[1]:
            idx_range = (idx_range[0],(idx_range[1]+n_subfolder-1))
        else:
            idx_range = (idx_range[0],(idx_range[1]-1))

        subfolder = f"{idx_range[0]}-{idx_range[1]}"
        return subfolder

    def divide_large_filecount(root_dir, image_set):

        valid_sets = {'train':'train_bbox.json','val':'val_bbox.json','test':'test_bbox.json'}
        if image_set in list(valid_sets):
            set_list = [image_set]
        elif image_set == 'all':
            set_list = list(valid_sets)
        else:
            Exception('Entered value for image set not valid')
            return
        for img_set in set_list:
            img_dir = os.path.join(root_dir,img_set)
            img_paths = glob.glob(os.path.join(img_dir,'*.jpg'))
            n_images = len(img_paths)
            print(f"found {n_images} images in {img_set} set")
            DataFiles.create_subfolders(img_dir,len(img_paths))
            for img_path in img_paths:
                name = (img_path.split('\\')[-1]).split('.')[0]
                idx = int(name.split('frame')[-1])
                DataFiles.move_file(idx, img_path, img_dir)
                print(f"moving {name}")

def plot_bbox(limits,axes):
    x= [limits[0],limits[0],limits[1],limits[1],limits[0]]
    y = [limits[2],limits[3],limits[3],limits[2],limits[2]]
    axes.plot(x,y,alpha = 0.6)


''' DATA SETTINGS'''

data_root = os.path.join(os.getcwd())#),'data')
dataset_name = ''   #'Envisat_Set1'
train_dir = os.path.join(data_root,dataset_name,'train')
test_dir = os.path.join(data_root,dataset_name,'test')
val_dir = os.path.join(data_root,dataset_name,'val')
model_dir = os.path.join(data_root,dataset_name,'3DFeaturePoints.csv')

img_size = (512,512)

'''need to make changes to data_format implmentation and get pose from file, when moving to the second dataset'''

kwargs= {
    'extension': 'jpg',
    'size':img_size,
    'datafile': {'train': 'data_train.txt','test': 'data_test.txt','val': 'data_val.txt'}, #datafile: generic txt file name in each folder
    'imgname_header': 'frame',
    'model': {'path': model_dir, 'unit':'cm'}, #3D model path
    'data_format' : {
        'id':None,
        'z_boresight': None,
        'attitude': None,
        'AzEl':None,
        'features': (0,32)
        },
    'feature_format':{'id_included':False},
    'n_subfolder':1000 # None if all images in one folder
    }

# -----------------------------------
''' Processsing'''


thiscam = Camera(img_size)
Envisat = ENVISAT_DS(camera_obj=thiscam, train_path= train_dir, val_path= val_dir, test_path= test_dir, **kwargs)

#Remove this when the pose data is in the same file as feature data- see kwargs instruction

#Envisat.get_pose_from_original_data(os.path.join(train_dir,'original_data.txt'),'train')
#Envisat.get_pose_from_original_data(os.path.join(val_dir,'original_data.txt'),'val')
#-----

## Write data to bbox JSON
Envisat.write2JSON(image_set='test',n_max =25000, filename='test_set_only_pose')


## Check Projection
#Envisat.check_projection('train',idx=0)
#Envisat.check_projection('val',idx=0)

#Move images to subfolders for Colab/Drive problem

#gt_data = Envisat.gt_data
#DataFiles.divide_large_filecount(os.path.join(data_root,dataset_name),'all')

# Envisat.visualize('train',idx=10)

# eul = [ 100,23,45]
# rad = np.radians(eul)
# quat = Kinematics.eul2quat(rad,'ZYX')

# You can also use COCO to access the COCO annotations files
results =  Envisat.load_HRNet_results('HRNet_regression/Results/Envisat_1/keypoints_val_results_0.json')
Envisat.visualize_result(results,'val',img_id=1546, **kwargs)