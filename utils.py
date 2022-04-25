# 获取图像格式
def suffix(name):
    for i in range(len(name)):
        if name[i] == '.':
            dot_index = i
    return str(name[dot_index+1:])

# 对文件排序
def SortFilename(filelist):
    maxlabel = max(filelist,key=filelist.count)
    index = filelist.index(maxlabel)
    suff = suffix(filelist[index])
    suf = len(suff)
    key = []
    for i in range(len(filelist)):
        if filelist[i][-suf:] == suff:
            temp = filelist[i][:-suf-1] 
            for j in range(1, len(temp)):
                if temp[-j].isdigit() == False:
                    key_temp = float(temp[-(j-1):])
                    key.append(key_temp)
                    break
        else:
            continue
            
    for i in range(len(key)):
        for j in range(len(key)-i-1):
            if key[j] > key[j+1]:
                key[j], key[j+1] = key[j+1], key[j]
                filelist[j], filelist[j+1] = filelist[j+1], filelist[j]
    return filelist


def fig2im(fig):
    '''
    matplotlib.figure.Figure转为np.ndarray
    '''
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype='u1')
    im = buf_ndarray.reshape(h, w, 3)
    return im


def tracked_label(x, y, im):
    fig = plt.figure(figsize=(5,5),dpi=400) 
    ax = fig.add_subplot(111)
    ax.imshow(im)
    for i in range(len(x)):
        coo_x = int(x[i])
        coo_y = int(y[i])
        ax.text(x[i], y[i], tracked_movie[0, coo_y, coo_x],color='red',fontsize=6)
    im = fig2im(fig)
    
    return im