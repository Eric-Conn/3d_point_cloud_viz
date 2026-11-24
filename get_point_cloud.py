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






### Point Cloud Primitives



def get_horizontal_ruled_sphere(num_circle_points):


    # num_circle_points = 50

    bar = np.arange(num_circle_points) / num_circle_points * 2*np.pi

    unit_circle = np.cos(bar)[:,None]*i[None , :] + np.sin(bar)[:,None]* j[None , :]

    elevation_angles = np.arange(0 , num_circle_points )
    elevation_angles = elevation_angles*np.pi*2 /num_circle_points

    radii = np.sin(elevation_angles)
    radii_heights = np.cos(elevation_angles)

    radii_heights_array = np.repeat(radii_heights[:,None]*k ,repeats = num_circle_points, axis = 0)

    radii_array = np.repeat(radii , repeats = num_circle_points , axis = 0)


    sphere = radii_heights_array + np.tile(unit_circle ,(num_circle_points,1))*radii_array[:,None]



    sphere = np.concat([sphere[num_circle_points: , :] , -sphere] , axis = 0 )




    # sphere = radii[1:,None] * unit_circle[None,: ,:] + radii_heights[:,None]*k
    return sphere

def get_horizontal_ruled_cylinder(num_circle_points):



    # num_circle_points = 50

    bar = np.arange(num_circle_points) / num_circle_points * 2*np.pi

    unit_circle = np.cos(bar)[:,None]*i[None , :] + np.sin(bar)[:,None]* j[None , :]

    elevation_angles = np.arange(0 , num_circle_points )
    elevation_angles = elevation_angles*np.pi*2 /num_circle_points

    radii = np.sin(elevation_angles)
    radii_heights = np.cos(elevation_angles)

    radii_heights_array = np.repeat(radii_heights[:,None]*k ,repeats = num_circle_points, axis = 0)

    radii_array = np.repeat(radii , repeats = num_circle_points , axis = 0)


    cylinder = radii_heights_array + np.tile(unit_circle ,(num_circle_points,1))



    cylinder = np.concat([cylinder[num_circle_points: , :] , -cylinder] , axis = 0 )




    # sphere = radii[1:,None] * unit_circle[None,: ,:] + radii_heights[:,None]*k
    return cylinder
