import numpy as np
import torch
import torch.nn.functional as F
import glob
import csv
import copy
from evaluate_imp_performance import func_eval_imp_perf
#import matplotlib.pyplot as plt
import random
import os

"""Assign an available gpu to this code, if any."""
gpu_id = ['0']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)


"""The pre-computed statistics of all 13 labs."""
NORM = np.array([103.60919172, 4.09204719,25.50716671,138.56716363,30.08871304,10.04918433,90.30916338,250.02270207,11.32065882,16.22554337,32.77929399,1.6120934,130.27666338])  # mean value
STD = np.array([6.14830006,0.60043011,5.00476506,4.94450959,4.76858583,1.63745679,6.56035119,166.09335688,7.55992873,2.46055167,25.19313938,1.64061308,59.0473318])   # standard deviation


CUT_RATIO = 0.9 # the ratio of data for training the regressor (out of the given data of the learning phase of the challenge) Under such ratio, data from 7445 patients are used for training, and 8267-7445 for validation. 


feature_model = [1,2,3,4,5,6,7,8,9,10,11,12,13]


class nRMSELoss(torch.nn.Module):
    """Define the loss function for model training based on the final evaluation metric."""
    def __init__(self):
        super(nRMSELoss, self).__init__()

    def forward(self, input, target, abs_extreme_diff):
        return torch.mean(torch.div(torch.pow((input - target),2), abs_extreme_diff))


class Net(torch.nn.Module):
    """Define the deep neural network for lab prediction task."""
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, n_hidden_6,n_hidden_7,n_hidden_8):
        super(Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden_1)   
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2) 
        self.hidden_3 = torch.nn.Linear(n_hidden_2, n_hidden_3)  
        self.hidden_4 = torch.nn.Linear(n_hidden_3, n_hidden_4)  
        self.hidden_5 = torch.nn.Linear(n_hidden_4, n_hidden_5)  
        self.hidden_6 = torch.nn.Linear(n_hidden_5, n_hidden_6)  
        self.hidden_7 = torch.nn.Linear(n_hidden_6, n_hidden_7)  
        self.hidden_8 = torch.nn.Linear(n_hidden_7, n_hidden_8) 
        self.out = torch.nn.Linear(n_hidden_8, 1) 
        self.bn_1 = torch.nn.BatchNorm1d(num_features=n_hidden_5)
    
    def forward(self, x): 
        x1 = F.relu(self.hidden_1(x))
        x1 = F.relu(self.hidden_2(x1))
        x1 = F.relu(self.hidden_3(x1))
        x1 = F.relu(self.hidden_4(x1))
        x1 = F.relu(self.bn_1(self.hidden_5(x1)+x))
        x1 = F.relu(self.hidden_6(x1))
        x1 = self.out(self.hidden_8(F.relu(self.hidden_7(x1))))

        return x1




patient_num = len(glob.glob1('./filled_groundtruth/mix_impute/', "*.csv"))
#print(patient_num)
miss_position_all = []
miss_NA_position_all = []
for patient_id in range(patient_num):
    file_name_miss = './filled_groundtruth/miss_marked/new' + str(patient_id+1) + '.csv'    # the csv files with 'MISS' and 'NA' marked (marked based on the groundtruth folder and train with missing folder)
    miss_position = []    # store the missing positions (which have groundtruth) across all patients
    miss_NA_position = []    # store the missing positions (which have groundtruth) and NA positions (which have no groundtruth)
    with open(file_name_miss,'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        head = True
        row_index = 0
        for row in reader:
            if head == False:    # need to remove the second condition
                lab_id_miss = [index for index in range(len(row)) if row[index] == 'MISS' and index in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
                lab_id_miss_NA = [index for index in range(len(row)) if (row[index] == 'MISS' or row[index] == 'NA') and index in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
                for lab in lab_id_miss:
                    miss_position.append([patient_id+1, row_index, lab])

                for lab in lab_id_miss_NA:
                    miss_NA_position.append([patient_id+1, row_index, lab])

                row_index = row_index + 1
            head = False
    miss_position_all.append(miss_position)
    miss_NA_position_all.append(miss_NA_position)
    #print(miss_position)

fill_result_all = []
fill_result_test = []
for patient_id in range(patient_num):    # if you want to test the performance of our model in both training and validation set, please use this line
#for patient_id in range(7445,8267):
    file_name_gr_tr = './filled_groundtruth/mix_impute/' + str(patient_id + 1) + '.csv'    # load the prefilled data.
    patient_data_filled = []
    with open(file_name_gr_tr,'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        head = True
        row_index = 0
        for row in reader:
            if head == False:
                temp_row = [float(row[b]) for b in range(len(row))]
                temp_row[0] = temp_row[0] / 60.0
                temp_row[1:]= list((temp_row[1:]-NORM)/STD)
                patient_data_filled.append(temp_row)    # include time stamp
            head = False
    patient_data_filled = copy.deepcopy(np.array(patient_data_filled))
    
    ####################################################### construct features from neighboring lab vectors
    need_fill = miss_NA_position_all[patient_id]
    for position in need_fill:
        row_need_fill = copy.deepcopy(patient_data_filled[position[1]])

        row_need_fill_appr = copy.deepcopy(row_need_fill)
        if position[1] == 0:
            row_need_fill_appr = np.delete(row_need_fill_appr, position[2])
            add_part = np.array([0.0, patient_data_filled[1][position[2]],0.0, patient_data_filled[1][position[2]], abs(patient_data_filled[1][0]-patient_data_filled[0][0]), patient_data_filled[1][position[2]],abs(patient_data_filled[2][0]-patient_data_filled[0][0]), patient_data_filled[2][position[2]]])
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part))
            add_part1 = np.array(list(patient_data_filled[1][1:])+list(patient_data_filled[1][1:])+list(patient_data_filled[1][1:])+list(patient_data_filled[2][1:]))
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part1))
        elif position[1] == 1:
            row_need_fill_appr = np.delete(row_need_fill_appr, position[2])
            add_part = np.array([abs(patient_data_filled[1][0]-patient_data_filled[0][0]), patient_data_filled[0][position[2]], abs(patient_data_filled[1][0]-patient_data_filled[0][0]), patient_data_filled[0][position[2]], abs(patient_data_filled[2][0]-patient_data_filled[1][0]), patient_data_filled[2][position[2]],abs(patient_data_filled[3][0]-patient_data_filled[1][0]), patient_data_filled[3][position[2]]])
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part))
            add_part1 = np.array(list(patient_data_filled[0][1:])+list(patient_data_filled[0][1:])+list(patient_data_filled[2][1:])+list(patient_data_filled[3][1:]))
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part1))
        elif position[1] == len(patient_data_filled) - 1:
            row_need_fill_appr = np.delete(row_need_fill_appr, position[2])
            add_part = np.array([abs(patient_data_filled[len(patient_data_filled) - 3][0]-patient_data_filled[len(patient_data_filled) - 1][0]),patient_data_filled[len(patient_data_filled) - 3][position[2]],abs(patient_data_filled[len(patient_data_filled) - 2][0]-patient_data_filled[len(patient_data_filled) - 1][0]),patient_data_filled[len(patient_data_filled) - 2][position[2]],0.0,patient_data_filled[len(patient_data_filled) - 2][position[2]],0.0,patient_data_filled[len(patient_data_filled) - 2][position[2]]])
            row_need_fill_appr = np.concatenate((row_need_fill_appr, add_part))
            add_part1 = np.array(list(patient_data_filled[len(patient_data_filled) - 3][1:])+list(patient_data_filled[len(patient_data_filled) - 2][1:])+list(patient_data_filled[len(patient_data_filled) - 2][1:])+list(patient_data_filled[len(patient_data_filled) - 2][1:]))
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part1))
        elif position[1] == len(patient_data_filled) - 2:
            row_need_fill_appr = np.delete(row_need_fill_appr, position[2])
            add_part = np.array([abs(patient_data_filled[len(patient_data_filled) - 4][0]-patient_data_filled[len(patient_data_filled) - 2][0]),patient_data_filled[len(patient_data_filled) - 4][position[2]],abs(patient_data_filled[len(patient_data_filled) - 3][0]-patient_data_filled[len(patient_data_filled) - 2][0]),patient_data_filled[len(patient_data_filled) - 3][position[2]],abs(patient_data_filled[len(patient_data_filled) -1][0]-patient_data_filled[len(patient_data_filled) - 2][0]),patient_data_filled[len(patient_data_filled) - 1][position[2]],abs(patient_data_filled[len(patient_data_filled) -1][0]-patient_data_filled[len(patient_data_filled) - 2][0]),patient_data_filled[len(patient_data_filled) - 1][position[2]]])
            row_need_fill_appr = np.concatenate((row_need_fill_appr, add_part))
            add_part1 = np.array(list(patient_data_filled[len(patient_data_filled) - 4][1:])+list(patient_data_filled[len(patient_data_filled) - 3][1:])+list(patient_data_filled[len(patient_data_filled) - 1][1:])+list(patient_data_filled[len(patient_data_filled) - 1][1:]))
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part1))
        else:
            row_need_fill_appr = np.delete(row_need_fill_appr, position[2])
            add_part = np.array([abs(patient_data_filled[position[1]-2][0]-patient_data_filled[position[1]][0]), patient_data_filled[position[1]-2][position[2]],abs(patient_data_filled[position[1]-1][0]-patient_data_filled[position[1]][0]), patient_data_filled[position[1]-1][position[2]], abs(patient_data_filled[position[1]+1][0]-patient_data_filled[position[1]][0]), patient_data_filled[position[1]+1][position[2]], abs(patient_data_filled[position[1]+2][0]-patient_data_filled[position[1]][0]), patient_data_filled[position[1]+2][position[2]]])
            row_need_fill_appr = np.concatenate((row_need_fill_appr, add_part))
            add_part1 = np.array(list(patient_data_filled[position[1]-2][1:])+list(patient_data_filled[position[1]-1][1:])+list(patient_data_filled[position[1]+1][1:])+list(patient_data_filled[position[1]+2][1:]))
            row_need_fill_appr = np.concatenate((row_need_fill_appr,add_part1))
        row_need_fill_list = list(row_need_fill_appr)

        PATH = './selected_model/selected_model_lab_' + str(feature_model[position[2] - 1]) + '.pkl'    # load the trained model for prediction

        if torch.cuda.is_available():
            model = torch.load(PATH)
            model = model.cuda()
        else:
            model = torch.load(PATH, map_location='cpu')

        model.eval()
        test_data = torch.from_numpy(np.reshape(np.array(row_need_fill_list[1:]),(1,-1))).type(torch.FloatTensor)
        if torch.cuda.is_available():
            test_data = test_data.cuda()
        out = model(test_data)

        patient_data_filled[position[1],position[2]] = (out.detach().cpu().numpy()[0] - NORM[position[2]-1]) / STD[position[2]-1]

        if [patient_id+1, position[1], position[2]] in miss_position_all[patient_id]:
            fill_result_all.append((patient_id + 1, (position[1], position[2]), out.detach().cpu().numpy()[0]))
            if patient_id >= 7444:
                fill_result_test.append((patient_id + 1, (position[1], position[2]), out.detach().cpu().numpy()[0]))

#np.save('fill_result_all.npy', fill_result_all)    # store the results of filling for all missings and NAs
#np.save('fill_result_test.npy', fill_result_test)    # store the results of filling for all missings
print("The performance of 13 labs over all patients in the training phase: ")
performance_all = func_eval_imp_perf(fill_result_all)
print(performance_all)
print("Average performance: %d" % (sum(performance_all)/float(len(performance_all))))

print("The performance of 13 labs over validation patients in the training phase: ")
performance_test = func_eval_imp_perf(fill_result_test)
print(performance_test)
print("Average performance: %d" % (sum(performance_test)/float(len(performance_test))))
