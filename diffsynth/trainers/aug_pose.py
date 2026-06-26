import os
import random
import numpy as np
import pickle

def load_keypoints_pkl(pkl_path):
    """读取形状 (f, 133, 3) 的pkl文件"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data  # (f,133,3)

def mp_main(pose, start, end, mode='train'):
    
    
    
    weight_pool = [round(0.6 + 0.1 * i, 1) for i in range(8)]
    body_scale = random.choice(weight_pool)
    arm_scale = random.choice(weight_pool)
    leg_scale = random.choice(weight_pool)
    neck_scale = random.choice(weight_pool)
    head_scale = random.choice(weight_pool)
    
    bodies = pose['bodies'][start]
    # faces = pose['faces'][0]
    hands = pose['hands'][start]
    candidate = bodies


    zero_2_x = candidate[2][0]
    zero_2_y = candidate[2][1]
    zero_5_x = candidate[5][0]
    zero_5_y = candidate[5][1]
    zero_8_x = candidate[8][0]
    zero_8_y = candidate[8][1]
    zero_11_x = candidate[11][0]
    zero_11_y = candidate[11][1]
    zero_center1 = 0.5*(candidate[2]+candidate[5])
    zero_center2 = 0.5*(candidate[8]+candidate[11])

    x_ratio = body_scale
    y_ratio = body_scale

    pose['bodies'][start][:,0] *= x_ratio
    pose['bodies'][start][:,1] *= y_ratio
    # pose['faces'][0][:,:,0] *= x_ratio
    # pose['faces'][0][:,:,1] *= y_ratio
    pose['hands'][start][:,:,0] *= x_ratio
    pose['hands'][start][:,:,1] *= y_ratio
    
    ########neck########
    neck_ratio = neck_scale

    x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
    y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

    pose['bodies'][start][0,0] += x_offset_neck
    pose['bodies'][start][0,1] += y_offset_neck
    pose['bodies'][start][14,0] += x_offset_neck
    pose['bodies'][start][14,1] += y_offset_neck
    pose['bodies'][start][15,0] += x_offset_neck
    pose['bodies'][start][15,1] += y_offset_neck
    pose['bodies'][start][16,0] += x_offset_neck
    pose['bodies'][start][16,1] += y_offset_neck
    pose['bodies'][start][17,0] += x_offset_neck
    pose['bodies'][start][17,1] += y_offset_neck
    
    ########shoulder2########

    shoulder2_ratio = body_scale

    x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
    y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

    pose['bodies'][start][2,0] += x_offset_shoulder2
    pose['bodies'][start][2,1] += y_offset_shoulder2
    pose['bodies'][start][3,0] += x_offset_shoulder2
    pose['bodies'][start][3,1] += y_offset_shoulder2
    pose['bodies'][start][4,0] += x_offset_shoulder2
    pose['bodies'][start][4,1] += y_offset_shoulder2
    pose['hands'][start][1,:,0] += x_offset_shoulder2
    pose['hands'][start][1,:,1] += y_offset_shoulder2

    ########shoulder5########

    shoulder5_ratio = body_scale

    x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
    y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

    pose['bodies'][start][5,0] += x_offset_shoulder5
    pose['bodies'][start][5,1] += y_offset_shoulder5
    pose['bodies'][start][6,0] += x_offset_shoulder5
    pose['bodies'][start][6,1] += y_offset_shoulder5
    pose['bodies'][start][7,0] += x_offset_shoulder5
    pose['bodies'][start][7,1] += y_offset_shoulder5
    pose['hands'][start][0,:,0] += x_offset_shoulder5
    pose['hands'][start][0,:,1] += y_offset_shoulder5

    ########arm3########

    arm3_ratio = arm_scale

    x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
    y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

    pose['bodies'][start][3,0] += x_offset_arm3
    pose['bodies'][start][3,1] += y_offset_arm3
    pose['bodies'][start][4,0] += x_offset_arm3
    pose['bodies'][start][4,1] += y_offset_arm3
    pose['hands'][start][1,:,0] += x_offset_arm3
    pose['hands'][start][1,:,1] += y_offset_arm3

    ########arm4########

    arm4_ratio = arm_scale

    x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
    y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

    pose['bodies'][start][4,0] += x_offset_arm4
    pose['bodies'][start][4,1] += y_offset_arm4
    pose['hands'][start][1,:,0] += x_offset_arm4
    pose['hands'][start][1,:,1] += y_offset_arm4

    ########arm6########

    arm6_ratio = arm_scale

    x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
    y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

    pose['bodies'][start][6,0] += x_offset_arm6
    pose['bodies'][start][6,1] += y_offset_arm6
    pose['bodies'][start][7,0] += x_offset_arm6
    pose['bodies'][start][7,1] += y_offset_arm6
    pose['hands'][start][0,:,0] += x_offset_arm6
    pose['hands'][start][0,:,1] += y_offset_arm6

    ########arm7########
    arm7_ratio = arm_scale

    x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
    y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

    pose['bodies'][start][7,0] += x_offset_arm7
    pose['bodies'][start][7,1] += y_offset_arm7
    pose['hands'][start][0,:,0] += x_offset_arm7
    pose['hands'][start][0,:,1] += y_offset_arm7

    ########head14########
    head14_ratio = head_scale

    x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
    y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

    pose['bodies'][start][14,0] += x_offset_head14
    pose['bodies'][start][14,1] += y_offset_head14
    pose['bodies'][start][16,0] += x_offset_head14
    pose['bodies'][start][16,1] += y_offset_head14

    ########head15########

    head15_ratio = head_scale

    x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
    y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

    pose['bodies'][start][15,0] += x_offset_head15
    pose['bodies'][start][15,1] += y_offset_head15
    pose['bodies'][start][17,0] += x_offset_head15
    pose['bodies'][start][17,1] += y_offset_head15

    ########head16########

    head16_ratio = head_scale

    x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
    y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

    pose['bodies'][start][16,0] += x_offset_head16
    pose['bodies'][start][16,1] += y_offset_head16

    ########head17########
    head17_ratio = head_scale

    x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
    y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

    pose['bodies'][start][17,0] += x_offset_head17
    pose['bodies'][start][17,1] += y_offset_head17
    
    ########MovingAverage########
    
    ########left leg########
    ll1_ratio = leg_scale

    x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
    y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

    pose['bodies'][start][9,0] += x_offset_ll1
    pose['bodies'][start][9,1] += y_offset_ll1
    pose['bodies'][start][10,0] += x_offset_ll1
    pose['bodies'][start][10,1] += y_offset_ll1
    pose['bodies'][start][19,0] += x_offset_ll1
    pose['bodies'][start][19,1] += y_offset_ll1

    ll2_ratio = leg_scale

    x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
    y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

    pose['bodies'][start][10,0] += x_offset_ll2
    pose['bodies'][start][10,1] += y_offset_ll2
    pose['bodies'][start][19,0] += x_offset_ll2
    pose['bodies'][start][19,1] += y_offset_ll2

    ########right leg########
    rl1_ratio = leg_scale

    x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
    y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

    pose['bodies'][start][12,0] += x_offset_rl1
    pose['bodies'][start][12,1] += y_offset_rl1
    pose['bodies'][start][13,0] += x_offset_rl1
    pose['bodies'][start][13,1] += y_offset_rl1
    pose['bodies'][start][18,0] += x_offset_rl1
    pose['bodies'][start][18,1] += y_offset_rl1

    rl2_ratio = leg_scale

    x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
    y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

    pose['bodies'][start][13,0] += x_offset_rl2
    pose['bodies'][start][13,1] += y_offset_rl2
    pose['bodies'][start][18,0] += x_offset_rl2
    pose['bodies'][start][18,1] += y_offset_rl2

    # offset = pose['bodies'][100][1] - pose['bodies'][0][1]
    x = np.random.uniform(-20, 20)
    y = np.random.uniform(-20, 20)
    H, W,_ = pose['size']
    offset = np.array([x / W, y / H])
    pose['bodies'][start] += offset[np.newaxis, :]
    # pose[0]['faces'] += offset[np.newaxis, np.newaxis, :]
    pose['hands'][start] += offset[np.newaxis, np.newaxis, :]
    num_assign = random.randint(0, 2)
    exclude = {0, 1, 2, 5, 8, 11}
    candidates = [i for i in range(20) if i not in exclude]
    
    drop_pose = random.sample(candidates, k=num_assign)  # 不重复抽样
    if drop_pose and mode=='train':
        pose['body_indices'][start][drop_pose] = -1
    else:
        pass
    
    for i in range(start+1, end):
        pose['bodies'][i][:,0] *= body_scale
        pose['bodies'][i][:,1] *= body_scale
        # pose['faces'][i][:,:,0] *= x_ratio
        # pose['faces'][i][:,:,1] *= y_ratio
        pose['hands'][i][:,:,0] *= body_scale
        pose['hands'][i][:,:,1] *= body_scale

        ########neck########
        x_offset_neck = (pose['bodies'][i][1][0]-pose['bodies'][i][0][0])*(1.-neck_ratio)
        y_offset_neck = (pose['bodies'][i][1][1]-pose['bodies'][i][0][1])*(1.-neck_ratio)

        pose['bodies'][i][0,0] += x_offset_neck
        pose['bodies'][i][0,1] += y_offset_neck
        pose['bodies'][i][14,0] += x_offset_neck
        pose['bodies'][i][14,1] += y_offset_neck
        pose['bodies'][i][15,0] += x_offset_neck
        pose['bodies'][i][15,1] += y_offset_neck
        pose['bodies'][i][16,0] += x_offset_neck
        pose['bodies'][i][16,1] += y_offset_neck
        pose['bodies'][i][17,0] += x_offset_neck
        pose['bodies'][i][17,1] += y_offset_neck

        ########shoulder2########
        

        x_offset_shoulder2 = (pose['bodies'][i][1][0]-pose['bodies'][i][2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (pose['bodies'][i][1][1]-pose['bodies'][i][2][1])*(1.-shoulder2_ratio)

        pose['bodies'][i][2,0] += x_offset_shoulder2
        pose['bodies'][i][2,1] += y_offset_shoulder2
        pose['bodies'][i][3,0] += x_offset_shoulder2
        pose['bodies'][i][3,1] += y_offset_shoulder2
        pose['bodies'][i][4,0] += x_offset_shoulder2
        pose['bodies'][i][4,1] += y_offset_shoulder2
        pose['hands'][i][1,:,0] += x_offset_shoulder2
        pose['hands'][i][1,:,1] += y_offset_shoulder2

        ########shoulder5########

        x_offset_shoulder5 = (pose['bodies'][i][1][0]-pose['bodies'][i][5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (pose['bodies'][i][1][1]-pose['bodies'][i][5][1])*(1.-shoulder5_ratio)

        pose['bodies'][i][5,0] += x_offset_shoulder5
        pose['bodies'][i][5,1] += y_offset_shoulder5
        pose['bodies'][i][6,0] += x_offset_shoulder5
        pose['bodies'][i][6,1] += y_offset_shoulder5
        pose['bodies'][i][7,0] += x_offset_shoulder5
        pose['bodies'][i][7,1] += y_offset_shoulder5
        pose['hands'][i][0,:,0] += x_offset_shoulder5
        pose['hands'][i][0,:,1] += y_offset_shoulder5

        ########arm3########

        x_offset_arm3 = (pose['bodies'][i][2][0]-pose['bodies'][i][3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (pose['bodies'][i][2][1]-pose['bodies'][i][3][1])*(1.-arm3_ratio)

        pose['bodies'][i][3,0] += x_offset_arm3
        pose['bodies'][i][3,1] += y_offset_arm3
        pose['bodies'][i][4,0] += x_offset_arm3
        pose['bodies'][i][4,1] += y_offset_arm3
        pose['hands'][i][1,:,0] += x_offset_arm3
        pose['hands'][i][1,:,1] += y_offset_arm3

        ########arm4########

        x_offset_arm4 = (pose['bodies'][i][3][0]-pose['bodies'][i][4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (pose['bodies'][i][3][1]-pose['bodies'][i][4][1])*(1.-arm4_ratio)

        pose['bodies'][i][4,0] += x_offset_arm4
        pose['bodies'][i][4,1] += y_offset_arm4
        pose['hands'][i][1,:,0] += x_offset_arm4
        pose['hands'][i][1,:,1] += y_offset_arm4

        ########arm6########

        x_offset_arm6 = (pose['bodies'][i][5][0]-pose['bodies'][i][6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (pose['bodies'][i][5][1]-pose['bodies'][i][6][1])*(1.-arm6_ratio)

        pose['bodies'][i][6,0] += x_offset_arm6
        pose['bodies'][i][6,1] += y_offset_arm6
        pose['bodies'][i][7,0] += x_offset_arm6
        pose['bodies'][i][7,1] += y_offset_arm6
        pose['hands'][i][0,:,0] += x_offset_arm6
        pose['hands'][i][0,:,1] += y_offset_arm6

        ########arm7########

        x_offset_arm7 = (pose['bodies'][i][6][0]-pose['bodies'][i][7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (pose['bodies'][i][6][1]-pose['bodies'][i][7][1])*(1.-arm7_ratio)

        pose['bodies'][i][7,0] += x_offset_arm7
        pose['bodies'][i][7,1] += y_offset_arm7
        pose['hands'][i][0,:,0] += x_offset_arm7
        pose['hands'][i][0,:,1] += y_offset_arm7

        ########head14########

        x_offset_head14 = (pose['bodies'][i][0][0]-pose['bodies'][i][14][0])*(1.-head14_ratio)
        y_offset_head14 = (pose['bodies'][i][0][1]-pose['bodies'][i][14][1])*(1.-head14_ratio)

        pose['bodies'][i][14,0] += x_offset_head14
        pose['bodies'][i][14,1] += y_offset_head14
        pose['bodies'][i][16,0] += x_offset_head14
        pose['bodies'][i][16,1] += y_offset_head14

        ########head15########

        x_offset_head15 = (pose['bodies'][i][0][0]-pose['bodies'][i][15][0])*(1.-head15_ratio)
        y_offset_head15 = (pose['bodies'][i][0][1]-pose['bodies'][i][15][1])*(1.-head15_ratio)

        pose['bodies'][i][15,0] += x_offset_head15
        pose['bodies'][i][15,1] += y_offset_head15
        pose['bodies'][i][17,0] += x_offset_head15
        pose['bodies'][i][17,1] += y_offset_head15

        ########head16########

        x_offset_head16 = (pose['bodies'][i][14][0]-pose['bodies'][i][16][0])*(1.-head16_ratio)
        y_offset_head16 = (pose['bodies'][i][14][1]-pose['bodies'][i][16][1])*(1.-head16_ratio)

        pose['bodies'][i][16,0] += x_offset_head16
        pose['bodies'][i][16,1] += y_offset_head16

        ########head17########
        x_offset_head17 = (pose['bodies'][i][15][0]-pose['bodies'][i][17][0])*(1.-head17_ratio)
        y_offset_head17 = (pose['bodies'][i][15][1]-pose['bodies'][i][17][1])*(1.-head17_ratio)

        pose['bodies'][i][17,0] += x_offset_head17
        pose['bodies'][i][17,1] += y_offset_head17

        # ########MovingAverage########

        ########left leg########
        x_offset_ll1 = (pose['bodies'][i][9][0]-pose['bodies'][i][8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (pose['bodies'][i][9][1]-pose['bodies'][i][8][1])*(ll1_ratio-1.)

        pose['bodies'][i][9,0] += x_offset_ll1
        pose['bodies'][i][9,1] += y_offset_ll1
        pose['bodies'][i][10,0] += x_offset_ll1
        pose['bodies'][i][10,1] += y_offset_ll1
        pose['bodies'][i][19,0] += x_offset_ll1
        pose['bodies'][i][19,1] += y_offset_ll1



        x_offset_ll2 = (pose['bodies'][i][10][0]-pose['bodies'][i][9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (pose['bodies'][i][10][1]-pose['bodies'][i][9][1])*(ll2_ratio-1.)

        pose['bodies'][i][10,0] += x_offset_ll2
        pose['bodies'][i][10,1] += y_offset_ll2
        pose['bodies'][i][19,0] += x_offset_ll2
        pose['bodies'][i][19,1] += y_offset_ll2

        ########right leg########

        x_offset_rl1 = (pose['bodies'][i][12][0]-pose['bodies'][i][11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (pose['bodies'][i][12][1]-pose['bodies'][i][11][1])*(rl1_ratio-1.)

        pose['bodies'][i][12,0] += x_offset_rl1
        pose['bodies'][i][12,1] += y_offset_rl1
        pose['bodies'][i][13,0] += x_offset_rl1
        pose['bodies'][i][13,1] += y_offset_rl1
        pose['bodies'][i][18,0] += x_offset_rl1
        pose['bodies'][i][18,1] += y_offset_rl1


        x_offset_rl2 = (pose['bodies'][i][13][0]-pose['bodies'][i][12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (pose['bodies'][i][13][1]-pose['bodies'][i][12][1])*(rl2_ratio-1.)

        pose['bodies'][i][13,0] += x_offset_rl2
        pose['bodies'][i][13,1] += y_offset_rl2
        pose['bodies'][i][18,0] += x_offset_rl2
        pose['bodies'][i][18,1] += y_offset_rl2

        pose['bodies'][i] += offset[np.newaxis, :]
        # pose['faces'][i] += offset[np.newaxis, np.newaxis, :]
        pose['hands'][i] += offset[np.newaxis, np.newaxis, :]
        if drop_pose and mode=='train':
            pose['body_indices'][i][drop_pose] = -1
        else:
            pass
    return pose
