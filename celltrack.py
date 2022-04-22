import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure,color
import math

class CellTracker:
    """
    input:movie must be segmented mask to be further tracked.
    """
    def __init__(self, 
                 var_threshold=0.8,
                opening_morphology_kwargs=None):
        self.var_threshold=var_threshold
        self.opening_morphology_kwargs = opening_morphology_kwargs
        
    def _get_bountries(self, movie):
        """
        return the bountries of every frame, the background is set to 255 and the boundtry line is set to 0
        """
        output = np.empty(movie.shape)
        if len(movie.shape) != 3:
            raise ValueError("Image dimension must be 3! ")
        else:
            for i in range(movie.shape[0]):
                output[i] = np.full(movie.shape[1:],255)
                for j in range(movie.shape[1]):
                    for k in range(movie.shape[2]):
                        if movie[i,j,k] < 2:
                            output[i,j,k] = 0
        return output 
        
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
        
        
    def _initial_label(self, bountry_movie):
        init_labeled_movie = []
        for i in range(bountry_movie.shape[0]):
            init_labeled_movie.append(measure.label(bountry_movie[i],connectivity=1))
        return init_labeled_movie
    
    def get_centroid(self, labeled_movie):
        """
        input：any labeled movie processed by skimage.measure.label
        output: coordinate(x, y) and area of every property of every frame.
        """
        x = []
        y = []
        area = []
        for i in range(len(labeled_movie)):
            x_stack, y_stack, area_stack = [], [], []
            for j in range(labeled_movie[i].max()):
                properties = measure.regionprops(labeled_movie[i])
                y_, x_ = properties[j].centroid
                area_ = properties[j].area
                x_stack.append(x_)
                y_stack.append(y_)
                area_stack.append(area_)
            x.append(x_stack)
            y.append(y_stack)
            area.append(area_stack)

        return x, y, area
    
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
        good_img = np.empty(temp.shape)
        for k in range(good_img.shape[0]):
            good_label = all_label[k]
            properties = measure.regionprops(init_labeled_movie[k])
            for j in range(len(properties)):
                dot_area = properties[j].coords
                for i in range(dot_area.shape[0]):
                    good_img[k,dot_area[i][0],dot_area[i][1]] = good_label[j]

        return good_img
    
    def track(self, 
              movie,
              #var_threshold=0.8,
              #opening_morphology_kwargs=None
             ):
              
        bountry_movie = movie
              
        if self.opening_morphology_kwargs == None:
            init_labeled_movie = self._initial_label(movie)
        else:
            bountry_movie = self._open_morphology(bountry_movie, self.opening_morphology_kwargs)
            init_labeled_movie = self._initial_label(bountry_movie)
        
        x, y, area = self.get_centroid(init_labeled_movie)
        distance_matrix = self._get_distance(x, y)
        row_min, row_min_index = self._get_min(distance_matrix)
        threshold = self._get_threshold(row_min, var_threshold=self.var_threshold)
        all_label = self._final_label(x, threshold, row_min, row_min_index)
              
        tracked_movie = self._fill_in(init_labeled_movie, all_label)
        
        return tracked_movie, zip(x, y, area, all_label)