#from equi_track.preprocess.filter import sig_filter
from Package_before_deployment.Preprocess.filter import sig_filter
import math
import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack

def rotangles(df, cutoff  ,Butter_filter = False, ds_fsctor=1):
    """
    The function calculate the pitch, roll, and yaw angles. 
    From the three angles it also give the rotation matrix 
    that rotate a vector readings from phone reference frame to global reference frame. 
    
    Args: 
      df ( dataframe): a data frame with acceleration, gyro, and magnetometer 
      cutoff (Hz): low pass filter cutoff  frequncy
      Butter_filter (bool): If true butter worth filter is computed
      df_factor: the decimation factor(the default is 1)

    Returns:
      separate one dimensional array for pitch, roll,and yaw angles(in radians) and acceleration in GRF

    """

    # initializing variabes (both in angle and degree)
    a_GRF = []
    pitch_all_rads = []
    roll_all_rads = []
    yaw_all_rads = []

    roll_angle_rad_all=[]
    pitch_angle_rad_all=[]
    yaw_angle_rad_all=[]

    roll_angle_deg_all = []
    pitch_angle_deg_all = []
    yaw_angle_deg_all = []
    av_ned_all = []
    roll_ned_all = []
    pitch_ned_all = []
    yaw_ned_all = []
    Rotation_matrix_all=[]
    acceleration_all_GRF = []
    parts =[2.1,2.2,3.1,3.2,3.3,3.4,4.1,4.2]
    parts = df.part.unique()
    data_all_horses_ = df.copy()
    for l in parts:
        #print(l)
        #print(parts)
        data_all_horses = data_all_horses_[data_all_horses_.part==l].reset_index(drop=True).copy()#data_all_horses_[data_all_horses_.part==l].reset_index(drop=True)
        #print(data_all_horses.shape)
        data_all_horses=data_all_horses.dropna(subset=['ax','ay','az','lax','lay','laz','avx','avy','avz'])

        if data_all_horses.shape[0]<10:
            continue
       
        #initializing temp variables
        roll_angle_rad = []
        roll_angle_deg = []
        sin_roll_angle = []
        tan_roll_angle = []
        cos_roll_angle =[]
    
        roll_ned = []
        pitch_ned = []
        yaw_ned = []
        av_ned =[]
        
        #Pitch angels
        pitch_angle_rad = []
        pitch_angle_deg = []
        sin_pitch_angle = []
        cos_pitch_angle = []

        yaw_angle_rad=[]
        yaw_angle_deg=[]
        acceleration_GRF = []

        #calculating gravity difference between raw and linear acceleration
        gx = data_all_horses.ax - data_all_horses.lax
        gy = data_all_horses.ay - data_all_horses.lay
        gz = data_all_horses.az - data_all_horses.laz
         
         #filtering the gravity (low pass filter)
        gx_filtered = sig_filter(gx, cutoff, Butter_filter=False, ds_fsctor =1)
        gy_filtered = sig_filter(gy, cutoff, Butter_filter=False, ds_fsctor =1 )
        gz_filtered = sig_filter(gz, cutoff, Butter_filter=False, ds_fsctor =1 )
 
        #calculatinh the gravity magnitude
        g_filtered_mag = ((gx_filtered**2)+(gy_filtered**2)+(gz_filtered**2))**0.5

        #down sampling gravity
        gx_filtered_ds = gx_filtered[0:len(gx_filtered)]
        gy_filtered_ds = gy_filtered[0:len(gx_filtered)]
        gz_filtered_ds = gz_filtered[0:len(gx_filtered)]

        #g_filtered_ds_mag = (( gx_filtered_ds**2 ) + (gy_filtered_ds**2) + (gz_filtered_ds**2))**0.5
        #filtering the magnetometer readings
        mx_filtered = sig_filter(data_all_horses.mx, cutoff, Butter_filter=False, ds_fsctor=1 ) 
        my_filtered = sig_filter(data_all_horses.my, cutoff, Butter_filter=False, ds_fsctor=1 ) 
        mz_filtered = sig_filter(data_all_horses.mz, cutoff, Butter_filter=False, ds_fsctor=1 )

        #down sampling filtered magnetometer reading
        mx_filtered_ds = mx_filtered[0:len(mx_filtered)]
        my_filtered_ds = my_filtered[0:len(my_filtered)]
        mz_filtered_ds = mz_filtered[0:len(mz_filtered)]

        #calculate the magnitude of magnetometer readings 
        filtered_m_mag = ((mx_filtered_ds**2)+(my_filtered_ds**2)+(mz_filtered_ds**2))**0.5

        #pichking the length for magnetometer, to make all columns equal length
        limit = min(len(mx_filtered_ds),len(my_filtered_ds),len(mz_filtered_ds)\
                     ,len(gx_filtered_ds),len(gy_filtered_ds),len(gz_filtered_ds))
    
        # make gravity vectors columns equal (with acceleration, magnetism , and angular velocity)
        gx_filtered_ds_subset = gx_filtered_ds[0:limit]
        gy_filtered_ds_subset = gy_filtered_ds[0:limit]
        gz_filtered_ds_subset = gz_filtered_ds[0:limit]
    
        #makes magnetometer readings equal
        mx_filtered_ds_subset = mx_filtered_ds[0:limit]
        my_filtered_ds_subset  =mx_filtered_ds[0:limit]
        mz_filtered_ds_subset =mx_filtered_ds[0:limit]
    
        #compute magnetometer readings magnitude
        g_ds_subset_mag = ((gx_filtered_ds_subset**2)+(gy_filtered_ds_subset**2)+(gz_filtered_ds_subset**2))**0.5        
        
        #filtering acceleration signals
        ax_filtered = sig_filter(data_all_horses.ax, cutoff, Butter_filter=False, ds_fsctor =1)
        ay_filtered = sig_filter(data_all_horses.ay, cutoff, Butter_filter=False, ds_fsctor =1 )
        az_filtered = sig_filter(data_all_horses.az, cutoff, Butter_filter=False, ds_fsctor =1 )
        
        #filtering linear acceleration signals
        lax_filtered_ds_subset = sig_filter(data_all_horses.lax, cutoff, Butter_filter=False, ds_fsctor =1)
        lay_filtered_ds_subset = sig_filter(data_all_horses.lay, cutoff, Butter_filter=False, ds_fsctor =1)
        laz_filtered_ds_subset = sig_filter(data_all_horses.laz, cutoff, Butter_filter=False, ds_fsctor =1)
        
        #filtering angular velocity signalnals
        avx_filtered_ds_subset = sig_filter(data_all_horses.avx, cutoff, Butter_filter=False, ds_fsctor =1)
        avy_filtered_ds_subset = sig_filter(data_all_horses.avy, cutoff, Butter_filter=False, ds_fsctor =1)
        avz_filtered_ds_subset = sig_filter(data_all_horses.avz, cutoff, Butter_filter=False, ds_fsctor =1)

        
        #making filtered acceleration equal
        ax_filtered_ds_subset = ax_filtered[0:limit]
        ay_filtered_ds_subset = ax_filtered[0:limit]
        az_filtered_ds_subset = ax_filtered[0:limit]    

        #computing the pitch angles
        for i in range(0,len(ax_filtered_ds_subset)):
            sin_pitch_angle.append(math.sin(gx_filtered_ds_subset[i]/g_ds_subset_mag[i]))
            cos_pitch_angle.append((1-(math.sin(gx_filtered_ds_subset[i]/g_ds_subset_mag[i]))**2)**0.5)
            pitch_angle_rad.append(math.asin(gx_filtered_ds_subset[i]/g_ds_subset_mag[i]))
            pitch_angle_deg.append(math.asin(gx_filtered_ds_subset[i]/g_ds_subset_mag[i])*180/np.pi) 

        #computing the roll angles
        for u in range(0,len(mx_filtered_ds_subset)):
            tan_roll_angle.append(gy_filtered_ds_subset[u]/gz_filtered_ds_subset[u])
            cos_roll_angle.append(gz_filtered_ds_subset[u]/(gz_filtered_ds_subset[u]**2+(gy_filtered_ds_subset[u])**2)**0.5)
            sin_roll_angle.append(gy_filtered_ds_subset[u]/(gz_filtered_ds_subset[u]**2+(gy_filtered_ds_subset[u])**2)**0.5)
            roll_angle_rad.append(math.atan2(gy_filtered_ds_subset[u],gz_filtered_ds_subset[u]))
            roll_angle_deg.append(math.atan2(gy_filtered_ds_subset[u],gz_filtered_ds_subset[u])*180/np.pi)           

        #computing the yaw angles         
        for j in range(0,len(mx_filtered_ds_subset)): 
            Rroll = np.array([[1,0,0],[0,cos_roll_angle[j],sin_roll_angle[j]],[0,-sin_roll_angle[j], cos_roll_angle[j]]])
            Rpitch = np.array([[cos_pitch_angle[j], 0, sin_pitch_angle[j]], [0,1,0],[-sin_pitch_angle[j], 0, cos_pitch_angle[j]]])
            C = np.dot(np.dot(Rroll.T,Rpitch.T),np.array([mx_filtered_ds_subset[j],my_filtered_ds_subset[j] , mz_filtered_ds_subset[j] ]) )
            yaw_angle_rad.append(math.atan(-C[1]/C[0]))
            yaw_angle_deg.append(math.atan(-C[1]/C[0])*180/np.pi)  

        #computing the rotation angle and calculate the acceleration in GRF
        for t in range(0,len(az_filtered_ds_subset)):
            Rroll = np.array([[1,0,0],[0,cos_roll_angle[t],sin_roll_angle[t]],[0,-sin_roll_angle[t], cos_roll_angle[t]]])
            Rpitch = np.array([[cos_pitch_angle[t], 0, sin_pitch_angle[t]], [0,1,0],[-sin_pitch_angle[t], 0, cos_pitch_angle[t]]])
            Ryaw = np.array([[math.cos(yaw_angle_rad[t]), math.sin(yaw_angle_rad[t]), 0],[-math.sin(yaw_angle_rad[t]),math.cos(yaw_angle_rad[t]),0],[0,0,1]])
            Rot_mat = np.dot(np.dot(Rroll ,Rpitch),Ryaw)
            #Rotation_matrix_all.append(Rot_mat)

            a_ = np.array([ax_filtered_ds_subset[t],ay_filtered_ds_subset[t],az_filtered_ds_subset[t]])
            #av_ = np.array([avx_filtered_ds_subset[t],avy_filtered_ds_subset[t],avz_filtered_ds_subset[t]])
              
            roll_ned.append(math.atan2(Rot_mat[2,1], Rot_mat[2,2]))
            pitch_ned.append(math.asin(Rot_mat[2,0]))
            yaw_ned.append(-math.atan2(Rot_mat[1,0], Rot_mat[0,0]))            
                        
            #find acceleration in GRF by multiplying the acceleration vector with the rotation matrix
            acceleration_GRF.append(np.dot(a_,Rot_mat))
            #av_ned.append(np.dot(av_,Rot_mat))

        #appending pitch, roll, yaw angles, and acceleration  from each horse/part
        roll_all_rads.append(roll_angle_rad)
        pitch_all_rads.append(pitch_angle_rad)
        yaw_all_rads.append(yaw_angle_rad)
        #av_ned_all.append(av_ned)
        acceleration_all_GRF.append(acceleration_GRF)


    return roll_all_rads, pitch_all_rads, yaw_all_rads,acceleration_all_GRF  #, av_ned_all#,roll_ned_all,pitch_ned_all,yaw_ned_all  #a_ned       

 