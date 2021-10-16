#!/usr/bin/env python
# coding: utf-8

# In[2]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

#TODO: write a correct function

def DeltaPhi(encoder_msg, prev_ticks):
    """
        Args:
            encoder_msg: ROS encoder message (ENUM)
            prev_ticks: Previous tick count from the encoders (int)
        Return:
            rotation_wheel: Rotation of the wheel in radians (double)
            ticks: current number of ticks (int)
    """
    
    N_tot = encoder_msg.resolution # number of ticks per wheel revolution
    ticks = encoder_msg.data # incremental count of ticks from the encoder
    
    # Evaluate the number of ticks since the last call 
    delta_ticks = ticks-prev_ticks
    alpha = 2*np.pi/N_tot # rotation per tick in radians 
    
    delta_phi = alpha*delta_ticks # in radians

    return delta_phi, ticks


# In[3]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your odometry function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

# TODO: write the odometry function

import numpy as np 

def poseEstimation( R, # radius of wheel (assumed identical) - this is fixed in simulation, and will be imported from your saved calibration for the physical robot
                    baseline_wheel2wheel, # distance from wheel to wheel; 2L of the theory
                    x_prev, # previous x estimate - assume given
                    y_prev, # previous y estimate - assume given
                    theta_prev, # previous orientation estimate - assume given
                    delta_phi_left, # left wheel rotation (rad)
                    delta_phi_right): # right wheel rotation (rad)
    
    """
        Calculate the current Duckiebot pose using the dead-reckoning approach.

        Returns x,y,theta current estimates:
            x_curr, y_curr, theta_curr (:double: values)
    """
    
    d_left = R * delta_phi_left 
    d_right = R * delta_phi_right
    
    theta_delta = (d_right - d_left)/baseline_wheel2wheel
    d_A = (d_left + d_right)/2
    x_delta = d_A * np.cos(theta_prev)
    y_delta = d_A * np.sin(theta_prev)
    
    
    # TODO: these are random values, you have to implement your own solution in here
    x_curr = x_prev + x_delta
    y_curr = y_prev + y_delta
    theta_curr = theta_prev + theta_delta

    return x_curr, y_curr, theta_curr

