import numpy as np
import torch
import torch.nn.functional as F
import glob
import csv
import copy
from evaluate_imp_performance import func_eval_imp_perf
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



data_extract = []    # extract train and validation dataset from given data  
column_min_max_diff = []    # record the diff between max and min values for each lab in each time step
target_appr_value_list_neigh = []    # record neighboring lab vectors for each extracted data vector
patient_num = 8267
record_neigh_labs = []    # record time diff between central time step and neighboring steps
for id in range(patient_num):

    ####################################################### extract data
    file_name = './train_groundtruth/' + str(id+1)+'.csv'
    row_included_list = []
    extracted_info_patient = []   # for computing the min max diff
    with open(file_name,'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        head = True

        row_index = -1
        for row in reader:
            if head == False and 'NA' not in row:
                row_new = [float(b) for b in row]
                row_new[0] = row_new[0] / 60.0
                data_extract.append(row_new)    # include the timestamp

                extracted_info_patient.append(row_new[1:])
                row_included_list.append(row_index)
            head = False
            row_index = row_index + 1
    extracted_info_patient = np.array(extracted_info_patient)
    max_array = np.max(extracted_info_patient,0)
    min_array = np.min(extracted_info_patient,0)
    diff = [list(max_array - min_array)] * len(row_included_list)
    column_min_max_diff = column_min_max_diff + diff

    ####################################################### prepare for feature enriching
    file_name_fill_ground = './filled_groundtruth/mix_impute/' + str(id+1)+'.csv'
    patient_data_filled = []
    with open(file_name_fill_ground,'rt') as csv_file:    # read the prefilled data out
        reader = csv.reader(csv_file, delimiter = ',')
        head = True
        for row in reader:
            if head == False:
                row_new = [float(b) for b in row]
                row_new[0] = row_new[0] / 60.0
                row_new[1:] = list((row_new[1:] - NORM)/STD)
                patient_data_filled.append(row_new)  # include the timestamp
            head = False

    ####################################################### construct features from neighboring lab vectors

    for need_index in row_included_list:
        subs_row = []
        if need_index == 0:
            for lab_index in feature_model:
                subs_row.append([0.0, patient_data_filled[1][lab_index],0.0, patient_data_filled[1][lab_index], abs(patient_data_filled[1][0]-patient_data_filled[0][0]), patient_data_filled[1][lab_index],abs(patient_data_filled[2][0]-patient_data_filled[0][0]),patient_data_filled[2][lab_index]])
            record_neigh_labs.append(patient_data_filled[1][1:]+patient_data_filled[1][1:]+patient_data_filled[1][1:]+patient_data_filled[2][1:])
        elif need_index == 1:
            for lab_index in feature_model:
                subs_row.append([abs(patient_data_filled[1][0]-patient_data_filled[0][0]),patient_data_filled[0][lab_index],abs(patient_data_filled[1][0]-patient_data_filled[0][0]), patient_data_filled[0][lab_index], abs(patient_data_filled[2][0]-patient_data_filled[1][0]), patient_data_filled[2][lab_index],abs(patient_data_filled[3][0]-patient_data_filled[1][0]),patient_data_filled[3][lab_index]])
            record_neigh_labs.append(patient_data_filled[0][1:]+patient_data_filled[0][1:]+patient_data_filled[2][1:]+patient_data_filled[3][1:])
        elif need_index == len(patient_data_filled) - 1:
            for lab_index in feature_model:
                subs_row.append([abs(patient_data_filled[len(patient_data_filled) - 3][0] - patient_data_filled[len(patient_data_filled) - 1][0]), patient_data_filled[len(patient_data_filled) - 3][lab_index],abs(patient_data_filled[len(patient_data_filled) - 2][0] - patient_data_filled[len(patient_data_filled) - 1][0]), patient_data_filled[len(patient_data_filled) - 2][lab_index], 0.0, patient_data_filled[len(patient_data_filled) - 2][lab_index], 0.0, patient_data_filled[len(patient_data_filled) - 2][lab_index]])
            record_neigh_labs.append(patient_data_filled[len(patient_data_filled) - 3][1:]+patient_data_filled[len(patient_data_filled) - 2][1:]+patient_data_filled[len(patient_data_filled) - 2][1:]+patient_data_filled[len(patient_data_filled) - 2][1:])
        elif need_index == len(patient_data_filled) - 2:
            for lab_index in feature_model:
                subs_row.append([abs(patient_data_filled[len(patient_data_filled) - 4][0] - patient_data_filled[len(patient_data_filled) - 2][0]), patient_data_filled[len(patient_data_filled) - 4][lab_index],abs(patient_data_filled[len(patient_data_filled) - 3][0] - patient_data_filled[len(patient_data_filled) - 2][0]), patient_data_filled[len(patient_data_filled) - 3][lab_index], abs(patient_data_filled[len(patient_data_filled) - 1][0] - patient_data_filled[len(patient_data_filled) - 2][0]), patient_data_filled[len(patient_data_filled) - 1][lab_index], abs(patient_data_filled[len(patient_data_filled) - 1][0] - patient_data_filled[len(patient_data_filled) - 2][0]), patient_data_filled[len(patient_data_filled) - 1][lab_index]])
            record_neigh_labs.append(patient_data_filled[len(patient_data_filled) - 4][1:]+patient_data_filled[len(patient_data_filled) - 3][1:]+patient_data_filled[len(patient_data_filled) - 1][1:]+patient_data_filled[len(patient_data_filled) - 1][1:])
        else:
            for lab_index in feature_model:
                subs_row.append([abs(patient_data_filled[need_index-2][0] - patient_data_filled[need_index][0]), patient_data_filled[need_index-2][lab_index],abs(patient_data_filled[need_index-1][0] - patient_data_filled[need_index][0]), patient_data_filled[need_index-1][lab_index], abs(patient_data_filled[need_index+1][0] - patient_data_filled[need_index][0]), patient_data_filled[need_index+1][lab_index], abs(patient_data_filled[need_index+2][0] - patient_data_filled[need_index][0]), patient_data_filled[need_index+2][lab_index]])
            record_neigh_labs.append(patient_data_filled[need_index-2][1:]+patient_data_filled[need_index-1][1:]+patient_data_filled[need_index+1][1:]+patient_data_filled[need_index+2][1:])
        target_appr_value_list_neigh.append(subs_row)
    #########################################################
    
target_appr_value_neigh_array = np.array(target_appr_value_list_neigh) 
data_extract = np.array(data_extract)[:,1:]
data_extract = (data_extract - NORM) / STD
column_min_max_diff = np.array(column_min_max_diff)
record_neigh_labs = np.array(record_neigh_labs)


####################################################### For each lab, train a distinct predition model
for model_id in feature_model:
    temp = copy.deepcopy(data_extract)    # temp has no time stamp
    cut_point = int(len(data_extract) * CUT_RATIO)
    train_y = torch.from_numpy(temp[0:cut_point,model_id-1]*STD[model_id-1]+NORM[model_id-1]).type(torch.FloatTensor)
    test_y = torch.from_numpy(temp[cut_point:,model_id-1]*STD[model_id-1]+NORM[model_id-1]).type(torch.FloatTensor)    # for checking the loss dynamics (in a separated validation set)
    train_y = torch.unsqueeze(train_y, 1)
    test_y = torch.unsqueeze(test_y, 1)

    temp = np.delete(temp,model_id-1,1)
    x = np.concatenate((temp, target_appr_value_neigh_array[:,model_id-1,:]), axis=1)
    x = np.concatenate((x,record_neigh_labs),axis=1)

    train_x = torch.from_numpy(x[0:cut_point,:]).type(torch.FloatTensor)
    test_x = torch.from_numpy(x[cut_point:,:]).type(torch.FloatTensor)     # for checking the loss dynamics (in a separated validation set)
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        test_x  = test_x.cuda()
        train_y = train_y.cuda()
        test_y  = test_y.cuda()

    train_diff_min_max = torch.unsqueeze(torch.from_numpy(1./(column_min_max_diff[0:cut_point,model_id-1]**2)).type(torch.FloatTensor), 1)
    test_diff_min_max = torch.unsqueeze(torch.from_numpy(1./(column_min_max_diff[cut_point:,model_id-1]**2)).type(torch.FloatTensor), 1)
    if torch.cuda.is_available():
        train_diff_min_max = train_diff_min_max.cuda()
    net = Net(n_feature=len(feature_model)+3+26+4+26, n_hidden_1=64, n_hidden_2=128, n_hidden_3=256, n_hidden_4=128, n_hidden_5=72, n_hidden_6=32,n_hidden_7=16, n_hidden_8=8)
    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    loss_func = torch.nn.MSELoss(reduce=False)

    loss_train = []
    loss_test = []
    loss_min_te = 10000000
    best_model = copy.deepcopy(net)
    for step in range(3000):
        net.train()
        #print(step)
        out = net(train_x)  # input x and predict based on x
        loss = torch.mean(loss_func(out, train_y)*train_diff_min_max)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients


        if step % 5 == 0:
            net.eval()
            out = net(test_x)
            loss_te = torch.mean(loss_func(out, test_y).cpu() * test_diff_min_max)    # compute the nRMSE loss
            #print(step)
            #print(loss_te)
            #print(loss)
            ll = loss_te.detach().numpy()
            if ll < loss_min_te:
                loss_min_te = ll
                best_model = copy.deepcopy(net)
    print("feature %d " % model_id, flush=True)
    print("Best loss in test: %f" % loss_min_te, flush=True)
    file_name = './model_' + str(model_id) +'.pkl'
    torch.save(best_model, file_name)    # save the model according to the performance (loss) on the separated validation set.














######################################################################################################################### Checking the filling performance of the learned models over the validation set 

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
#for patient_id in range(patient_num):    # if you want to test the performance of our model in both training and validation set, please use this line
for patient_id in range(7445,8267):
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

        PATH = './model_' + str(feature_model[position[2] - 1]) + '.pkl'    # load the trained model for prediction
        model = torch.load(PATH)
        if torch.cuda.is_available():
            model = model.cuda()
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
np.save('fill_result_test.npy', fill_result_test)    # store the results of filling for all missings

#print("The performance of 13 labs over all patients in the training phase: ")
#performance_all = func_eval_imp_perf(fill_result_all)
#print(performance_all)
#print("Average performance: %d" % (sum(performance_all)/len(performance_all)))

print("The performance of 13 labs over validation patients in the training phase: ")
performance_test = func_eval_imp_perf(fill_result_test)
print(performance_test)
print("Average performance: %d" % (sum(performance_test)/float(len(performance_test))))
