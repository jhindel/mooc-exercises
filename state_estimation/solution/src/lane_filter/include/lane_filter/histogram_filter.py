#!/usr/bin/env python
# coding: utf-8

# In[1]:


# start by importing some things we will need
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy, multivariate_normal
from math import floor, sqrt


# In[4]:


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief


# In[5]:


# Now let's define the predict function


def histogram_predict(belief, dt, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
        belief_in = belief
        delta_t = dt
        
        # calculate v and w from ticks using kinematics. You will need `robot_spec`
        alpha = 2 * np.pi / robot_spec["encoder_resolution"] # wheel rotation per tick in radians
        d_left = robot_spec["wheel_radius"] * alpha * left_encoder_ticks # left wheel
        d_right = robot_spec["wheel_radius"] * alpha * right_encoder_ticks # right wheel
        
        v = (d_left + d_right)/2
        w = (d_right-d_left)/robot_spec["wheel_baseline"]
        
        # propagate each centroid forward using the kinematic function
        d_t = grid_spec['d'] + v * np.sin(grid_spec['phi'])
        phi_t = grid_spec['phi'] + w
        
        # print("d_t", d_t, "phi_t", phi_t)

        p_belief = np.zeros(belief.shape)

        # Accumulate the mass for each cell as a result of the propagation step
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                # If belief[i,j] there was no mass to move in the first place
                if belief[i, j] > 0:
                    # Now check that the centroid of the cell wasn't propagated out of the allowable range
                    if (
                        d_t[i, j] > grid_spec['d_max']
                        or d_t[i, j] < grid_spec['d_min']
                        or phi_t[i, j] < grid_spec['phi_min']
                        or phi_t[i, j] > grid_spec['phi_max']
                    ):
                        continue
                    
                    # Now find the cell where the new mass should be added
                    # replace with something that accounts for the movement of the robot
                    i_new = int(((d_t[i, j] - grid_spec["d_min"])/
                                 (grid_spec["d_max"] - grid_spec["d_min"]))*belief.shape[0])
                    j_new = int(((phi_t[i, j] - grid_spec["phi_min"])/
                                 (grid_spec["phi_max"] - grid_spec["phi_min"]))*belief.shape[1])
                    # print("i_new", i_new, "j_new", j_new)

                    p_belief[i_new, j_new] += belief[i, j]

        # Finally we are going to add some "noise" according to the process model noise
        # This is implemented as a Gaussian blur
        s_belief = np.zeros(belief.shape)
        gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

        if np.sum(s_belief) == 0:
            return belief_in
        belief = s_belief / np.sum(s_belief)
        return belief


# In[6]:


# We will start by doing a little bit of processing on the segments to remove anything that is behind the robot (why would it be behind?)
# or a color not equal to yellow or white

def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments


# In[7]:


def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane
            d_i = -d_i
            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i


# In[8]:


def generate_measurement_likelihood(segments, road_spec, grid_spec):

    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec['d'].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)
        
        # print(d_i, grid_spec['d_min'], phi_i, grid_spec['phi_min'])

        # if the vote lands outside of the histogram discard it
        if d_i > grid_spec['d_max'] or d_i < grid_spec['d_min'] or phi_i < grid_spec['phi_min'] or phi_i > grid_spec['phi_max']:
            continue
            

        #find the cell index that corresponds to the measurement d_i, phi_i
        i = int((d_i-grid_spec['d_min'])//grid_spec['delta_d'])
        j = int((phi_i-grid_spec['phi_min'])//grid_spec['delta_phi'])
        
        # print(i, j, measurement_likelihood.shape)
        
        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


# In[9]:


def softmax(x):
    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)
    
    # Finally, the update() function takes a measurement and uses it to update the weights of the histogram bins 
    # based on the incoming measurement,  𝑧𝑡
    # This is achieved by multiplying the weight in each bin by the likelihood that 
    # the measurement was generated by the state corresponding to the centroid of that bin.

    if measurement_likelihood is not None:
        # combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid probability distribution
        # max_measure = np.max(measurement_likelihood)
        # min_measure = np.min(measurement_likelihood)
        # norm_measurement_likelihood = (measurement_likelihood - min_measure)/(max_measure - min_measure)
        new_belief = belief*measurement_likelihood # combines the belief and the measurement_likelihood
        
        s = new_belief.sum()
        
        if s > 0:
            new_belief /= s
        else:
            new_belief = (belief*measurement_likelihood)/2
        
    return (measurement_likelihood, new_belief)

