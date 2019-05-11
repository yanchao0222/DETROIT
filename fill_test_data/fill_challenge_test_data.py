import numpy as np
import torch
import torch.nn.functional as F
import glob
import csv
import copy
import os

"""Assign an available gpu to this code, if any."""
gpu_id = ['0']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)


"""The pre-computed statistics of all 13 labs."""
NORM = np.array([103.60919172, 4.09204719,25.50716671,138.56716363,30.08871304,10.04918433,90.30916338,250.02270207,11.32065882,16.22554337,32.77929399,1.6120934,130.27666338])  # mean value
STD = np.array([6.14830006,0.60043011,5.00476506,4.94450959,4.76858583,1.63745679,6.56035119,166.09335688,7.55992873,2.46055167,25.19313938,1.64061308,59.0473318])   # standard deviation
feature_model = [1,2,3,4,5,6,7,8,9,10,11,12,13]


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

patient_num = len(glob.glob1('./challenge_test_data/', "*.csv"))
miss_position_all = []
for patient_id in range(patient_num):
    file_name_miss = './challenge_test_data/' + str(patient_id+1) + '.csv'    # the csv files with 'NA' marked
    miss_position = []    # store the missing positions across all patients
    with open(file_name_miss,'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        head = True
        row_index = 0
        for row in reader:
            if head == False:    # need to remove the second condition
                lab_id_miss = [index for index in range(len(row)) if row[index] == 'NA' and index in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
                for lab in lab_id_miss:
                    miss_position.append([patient_id+1, row_index, lab])
                row_index = row_index + 1
            head = False
    miss_position_all.append(miss_position)

fill_result_all = []
for patient_id in range(patient_num):
    file_name_gr_tr = './test_mix_impute/' + str(patient_id + 1) + '.csv'    # load the prefilled data.
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
    need_fill = miss_position_all[patient_id]
    filled = []
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
        fill_result_all.append((patient_id + 1, (position[1], position[2]), out.detach().cpu().numpy()[0]))
        filled.append((patient_id + 1, (position[1], position[2]), out.detach().cpu().numpy()[0]))
        patient_data_filled[position[1],position[2]] = (out.detach().cpu().numpy()[0] - NORM[position[2]-1]) / STD[position[2]-1]

    extracted = []
    file_name = './challenge_test_data/' + str(patient_id + 1) + '.csv'
    with open(file_name,'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        for row in reader:
            extracted.append(row)

    for posit in filled:
        extracted[posit[1][0]+1][posit[1][1]] = str(posit[2][0])

    save_name = './filled_data_for_evaluation/' + str(patient_id + 1) + '.csv'
    with open(save_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(extracted)
    csvfile.close()


np.save('fill_result_all.npy', fill_result_all)    # store the results of filling
print("Filling completed!")

