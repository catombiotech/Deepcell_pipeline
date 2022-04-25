import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import math
import sys
import pandas as pd
class CellTracker:
    """
    input:movie must be segmented mask to be further tracked.
    """
    def __init__(self, 
                 var_threshold=0.8,
                 threshold_picking_methods='otsu',
                opening_morphology_kwargs=None):
        self.var_threshold=var_threshold
        self.opening_morphology_kwargs = opening_morphology_kwargs
        self.threshold_picking_methods = threshold_picking_methods
    
    def _open_morphology(self, bountry_movie, opening_morphology_kwargs):
        """
        To perform opening operation on every images:
        bountry_movie: movie performed by _get_bountries(self, movie)
        kernel_size: the size of erode/dilate kernel, prefer (3,3) or (4,4).
        erode_iter_number: the number of eroding.
        dilate_iter_number: the number of dilating.
        """
        kernel_size = opening_morphology_kwargs['kernel_size']
        erode_iter_number = opening_morphology_kwargs['erode_iter_number']
        dilate_iter_number = opening_morphology_kwargs['dilate_iter_number']
        
        kernel = np.ones(kernel_size, np.uint8)
        for frame in bountry_movie:
            frame = frame.astype('uint8')
            frame = cv2.erode(frame, kernel, iterations=erode_iter_number)
            frame = cv2.dilate(frame, kernel, iterations=dilate_iter_number)
        
        return bountry_movie
        
    def _get_frame_centroid(self, frame):
        properties = measure.regionprops(frame)
        x = []
        y = []
        area = []

        for j in range(len(properties)): 
            y_, x_ = properties[j].centroid
            area_ = properties[j].area
            x.append(x_)
            y.append(y_)
            area.append(area_)
        return x, y, area
    
    def _get_movie_centroid(self, movie):
        x_all, y_all, area_all = [], [], []
        j = 0
        for frame in movie:
            x, y, area = self._get_frame_centroid(frame)
            x_all.append(x)
            y_all.append(y)
            area_all.append(area)
            sys.stdout.write('\r' + str(j+1) + " / " + str(len(movie)) + " regionprops extracted")
            sys.stdout.flush()
            j += 1
        return x_all, y_all, area_all

    
    def Distance(self, dot1, dot2):
        return math.sqrt(pow(dot1[0]-dot2[0],2)+pow(dot1[1]-dot2[1],2))
    
    def _get_distance(self, x, y):
        """
        return distance matrix.
        """
        distance = []
        for i in range(len(x)-1):
            dot_now, dot_pre = [], []
            x_now = x[i+1]
            y_now = y[i+1]
            x_pre = x[i]
            y_pre = y[i]
            for j in range(len(x_now)):
                dot_now.append((x_now[j],y_now[j]))
            for j in range(len(x_pre)):
                dot_pre.append((x_pre[j],y_pre[j]))

            distance_matrix = []
            for j in range(len(x_now)):
                distance_row = []
                for k in range(len(x_pre)):
                    distance_row.append(self.Distance(dot_now[j],dot_pre[k]))
                distance_matrix.append(distance_row)    
            distance.append(distance_matrix)
            
            sys.stdout.write('\r' + str(i+1) + " / " + str(len(x)-1) + " distanct matrix computed.")
            sys.stdout.flush()
        return distance
    
    def _get_min(self, distance):
        row_min = []
        row_min_index = []
        for j in range(len(distance)):
            matrix = distance[j]
            row_min_frame = []
            row_min_index_frame = []
            for i in range(len(matrix)):       
                min_value = min(matrix[i])
                min_index = matrix[i].index(min_value)
                row_min_frame.append(min_value)
                row_min_index_frame.append(min_index)
            row_min.append(row_min_frame)
            row_min_index.append(row_min_index_frame)

        return row_min, row_min_index
    
    def _get_threshold(self, row_min, var_threshold):
        threshold = []
        for frame in range(len(row_min)):
            data = np.copy(row_min[frame])
            var_list = []
            sorted_data = sorted(data)
            for i in range(len(data)):
                var_temp = np.var(sorted_data)
                var_list.append(var_temp)
                sorted_data.pop()
            delta_var = [var_list[i]-var_list[i+1] for i in range(len(var_list)-1)]
            for i in range(len(delta_var)):
                if delta_var[i] < var_threshold:   # here
                    big_idx = i
                    break
            sorted_data = sorted(data)
            threshold_frame = sorted_data[-big_idx]
            threshold.append(threshold_frame)
        return threshold
    
    def _otsu(self, row_frame_min):
        data = np.sort(row_frame_min)
        value, counts = np.unique(row_frame_min, return_counts=True)

        n = len(row_frame_min)
        u = np.mean(row_frame_min)
        delta_square = np.empty((n, ))

        for i in range(len(value)-1):
            n0 = (i+1)
            n1 = n - n0
            u0 = np.sum(value[:i+1]*counts[:i+1])/n0
            u1 = np.sum(value[i+1:]*counts[i+1:])/n1
            d = n0*(u0-u)**2 + n1*(u1-u)**2
            delta_square[i] = d
        index = np.where(delta_square==np.max(delta_square,axis=0)) 

        return value[index[0][0]-1]
    
    def _get_movie_otsu_thres(self, row_min):
        thres_list = []
        for min_list in row_min:
            thres = self._otsu(min_list)
            thres_list.append(thres)
        return thres_list
    
    def _final_label(self, x, threshold, row_min, row_min_index):
        all_label = []
        label_frame_first = np.arange(0,len(x[0]),1)
        label_frame_first = list(label_frame_first)
        max_label = len(x[0])-1
        all_label.append(label_frame_first)

        for i in range(len(row_min)):
            label_next = []
            thres_frame = threshold[i]
            min_value = row_min[i]
            for j in range(len(min_value)):
                if min_value[j] < thres_frame:
                    index_temp = row_min_index[i][j]
                    label_temp = all_label[i][index_temp]
                    label_next.append(label_temp)
                else:
                    max_label += 1
                    label_next.append(max_label)
            all_label.append(label_next)

        return all_label
    
    def _fill_in(self, init_labeled_movie, all_label):
        temp = np.copy(init_labeled_movie)
        good_img = np.zeros(temp.shape)
        for k in range(good_img.shape[0]):
            good_label = all_label[k]
            properties = measure.regionprops(init_labeled_movie[k])
            for j in range(len(properties)):
                dot_area = properties[j].coords
                for i in range(dot_area.shape[0]):
                    good_img[k,dot_area[i][0],dot_area[i][1]] = good_label[j]

        return good_img
    
    def track(self, movie):
        
        if not self.opening_morphology_kwargs == None:
            movie = self._open_morphology(movie, opening_morphology_kwargs=self.opening_morphology_kwargs)
            
   
        x,y,area = self._get_movie_centroid(movie)
        
        distance_matrix = self._get_distance(x, y)
        row_min, row_min_index = self._get_min(distance_matrix)
        #if self.threshold_picking_methods != 'ostu':
         #   threshold = self._get_threshold(row_min, var_threshold=self.var_threshold)
        #else:
        threshold = self._get_movie_otsu_thres(row_min)
            
        all_label = self._final_label(x, threshold, row_min, row_min_index)
              
        tracked_movie = self._fill_in(movie, all_label)
        
        return tracked_movie, zip(x, y, area, all_label)
