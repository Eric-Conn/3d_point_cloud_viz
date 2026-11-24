# This is our 3d visualizer function


import numpy as np
import numpy.linalg as lg

import matplotlib.pyplot as plt

import matplotlib.animation as animation


#World fixed basis

i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])




i_hom = np.array([1,0,0,1])
j_hom = np.array([0,1,0,1])
k_hom = np.array([0,0,1,1])







### cross matrix, rotation matrix, and affine transformation

def get_cross_mat(u):

    u = u/lg.norm(u)


    ux = np.array([
                    [0,-u[2] , u[1]],
                   [u[2] , 0 , -u[0] ] ,
                    [-u[1],u[0] , 0]
        ])


    return ux


def get_rot_matrix(u,theta):

    ux = get_cross_mat(u)

    rot = np.cos(theta)*np.eye(3) + (1 - np.cos(theta))*u[:,None]@u[None,:] + np.sin(theta)*ux

    return rot


def get_T(u,theta , t):

    R = get_rot_matrix(u,theta)

    T = np.block([[R , t[:,None]],
         [np.zeros((1 , 3)) , np.ones((1,1)) ]])

    return T



def get_T_inv(T):

    T1 = np.zeros(T.shape)
    T1[3,3] = 1


    T1[:3,:3] = T[:3,:3].T

    T1[:3 , 3] = -T1[:3,:3]@T[:3,3]

    return T1



###### Flattens from point cloud to 2d image data



# convention: our camera will be aimed along the j axis in the right handed camera frame.

def X_image(X,T):

    if X.shape[1] == 4:
        raise Exception('Do not input hom coors')



    # only local
    X_hom = np.concat([X , np.ones((X.shape[0],1))] , axis = 1)

    #Getting the inverse of the active camera motion to
    # generate passive coordinate transform

    T_inv = get_T_inv(T)


    #New coordinates of X

    #only local
    X_hom = X_hom@T_inv.T
    #transform to left handed camera coordinate system:

    # j direction becomes z axis

    #only local
    X_cam = X_hom[:, [0,2 , 1]]
    #only local
    mask = X_cam@k > 0

    # only local
    X_cam = X_cam[mask,:]





    zs = X_cam[:,2]

    im_X = X_cam / zs[:,None]




    im_X = im_X[:,:2]

    mask_box = lg.norm(im_X , ord = np.inf,axis = 1) <= 1

    im_X = im_X[mask_box , :]


    zs = zs[mask_box]

    sizes = 1/zs**2

    return im_X , sizes




#Takes the 2d image of the point cloud and displays it using a matplotlib scatterplot

def diplay_image_X(X_im , sizes ,T, plane = None):

    fig , ax = plt.subplots()



    if plane is not None:

        xy_plane = get_xy_plane()

        mask0 = lg.norm(xy_plane , axis =1,ord = 2 ) < 5
        xy_plane_masked = xy_plane[mask0]





        plane , plane_sizes = X_image(xy_plane_masked,T)



        ax.scatter(plane[:,0] , plane[:,1], c = 'skyblue',s = plane_sizes*10)





    ax.scatter(X_im[:,0] , X_im[:,1],s = sizes*10)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect(1)
    plt.show()





#######




#we wnat to define T based on where the j axis points and the location of the origin.

def camera_T(t , cam_j,roll = 0):

    cam_j = cam_j / lg.norm(cam_j)


    #Rotation from j to principle axis
    u = np.cross(j , cam_j)

    if np.array_equal(u ,i*0) :

        d = np.dot(j , cam_j)


        if d > 0:
            return get_T(k,0,t)
        else:
            return get_T(k,np.pi,t)


    theta =  np.acos(j@cam_j)



    T = get_T(u,theta,t)

    #This gives T, rotating from j to priciple axis



    return T




def get_xy_plane():


    cross = (np.repeat(np.arange(1,10) , 2 , axis = 0)[:,None] * np.tile(np.eye(2) , (9 , 1)) )
    cross = np.concat([cross , - cross] , axis = 0)/10


    xy_plane = np.mgrid[0:201 ,0:201]

    # xy_plane.shape

    xy_plane = xy_plane.reshape((2,-1) , order = 'F').T
    grid = np.repeat(xy_plane[:,:2] , 36 , axis = 0) + np.tile(cross , (xy_plane.shape[0],1))
    grid = np.concat([grid , xy_plane])
    xy_plane = grid

    xy_plane = (xy_plane - 100)
    xy_plane = np.concat([xy_plane , np.zeros(xy_plane.shape[0])[:,None]] , axis = 1)

    return xy_plane



def show_image(X,t,p):
    T = camera_T(t,p)
    X_im , sizes = X_image(X,T)
    diplay_image_X(X_im , sizes ,T, plane = True)







