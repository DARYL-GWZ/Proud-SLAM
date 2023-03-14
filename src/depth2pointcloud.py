import numpy as np

def rgbmap(rgb):
    rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
    return rgb

def pixel2cam(p, c_m): #像素坐标转相机归一化坐标
    return np.array([[(p[0] - c_m[0, 2]) / c_m[0, 0]], [(p[1] - c_m[1, 2]) / c_m[1, 1]]])

def depth_to_3dpoints(pose1,rgb,depth,K):
    pts_3d = []
    color_3d = []
    rgb = rgbmap(rgb)
    # print('rgb',rgb.shape)
    for m in range(0,depth.shape[0],3):
        for n in range(0,depth.shape[1],3):
            d = depth[m, n] 
            key_points = [n, m]
            # if d < 0.01 or d>10:
            if d < 0.01 :
                continue
            dd = d   #深度越大移动越大
            p1 = pixel2cam(key_points, K)
            p3d= [p1[0, 0]*dd, p1[1, 0]*dd, dd]
            c3d =[rgb[m,n,2],rgb[m,n,1],rgb[m,n,0]]
            #     p3d= np.append(p3d,[1])
            #     a = copy.deepcopy(p3d[0]) 
            #     b = copy.deepcopy(p3d[1])
            #     c = copy.deepcopy(p3d[2])
            #     p3d[0] = a  #正向向左
            #     p3d[1] = -b  #正向向上
            #     p3d[2] = -c #正向向后
            p3d= pose1[:3,:4] @ p3d
            color_3d.append(c3d)
            pts_3d.append(p3d)
    pts_3d = np.array(pts_3d)
    color_3d = np.array(color_3d)
    return pts_3d,color_3d

