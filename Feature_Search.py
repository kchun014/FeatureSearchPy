# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:53:37 2018

@author: Kau
"""
import numpy as np
import copy
import time
#===================================
#         Distance Function         
#===================================
def distance(x, y, p): #get euclidean distance
    totalDistance = 0
    for i,j in zip(x,y):
        newDistance = (abs(i-j)**p)
        totalDistance += newDistance
    return (totalDistance**(1/p))
#===================================
#     Cross Validation Function         
#===================================
def cross_validation(data, current_set, value, optimal): #value = current feature you're evaluating
    accuracy = 0
    least_wrong = 200
    labels = data[:, 0]
    features = data[:, 1:]
    wrong = 0
#    for each row:
#    Check against each other row
    for i in range(np.size(data, 0)):
        dist_best = 100000
        data_best = 100000
        for j in range(np.size(data, 0)):
           if i != j:
#                   Find closest neighbor
               dist = distance(features[i][current_set], features[j][current_set], 2) #give a row, and current features in set, to nearest_neighbor
#                   When done, assign label to neighbor's label
#                   Check own label, if not accurate, increment wrong counter.
               if dist < dist_best:
                   dist_best = dist
                   data_best = j
        correct = labels[i]
        guess = labels[data_best]
        if correct != guess:
            wrong += 1
            if optimal and wrong > least_wrong:
                return 0 #end early if not worth exploring.
        if i == (np.size(data, 0) - 1) and wrong < least_wrong:
            least_wrong = wrong
#    When done, assign label to neighbor's label
#    Check own label, if not accurate, increment wrong counter.
#    check against other rows, get accuracy. 
    accuracy = ((np.size(data, 0) -1) - wrong)/(np.size(data,0)-1)
    return accuracy
#===================================
#     Feature Search FunctionS         
#===================================
def feature_search_forward(data):
    current_set = []
    best_set = []
    best_ever_acc = 0
    for i in range(np.size(data, 1)-1):
        print('On the ' + str(i), 'th level of the search tree')
        feature_to_add = 0;
        best_so_far_acc = 0.0;
        for k in range(np.size(data, 1)-1):
            if k not in current_set:
                tempset = copy.deepcopy(current_set)
                tempset.append(k)
                print('--Considering adding the ', str(k+1) ,' feature')
                accuracy = cross_validation(data, tempset, k + 1, False) #pass in data, feature to add, index of current feature
                
                if accuracy > best_so_far_acc:
                    best_so_far_acc = accuracy;
                    feature_to_add = k;    
        current_set.append(feature_to_add)
        if best_ever_acc < best_so_far_acc:
            best_set = copy.deepcopy(current_set)
            best_ever_acc = best_so_far_acc
        print('On level ', str(i+1),', I added feature ', str(feature_to_add+1), ' to current set, with accuracy: ' + str(best_so_far_acc))
    return best_set, best_ever_acc
    
def feature_search_backward(data):
    current_set = list(range(np.size(data, 1)-1))
    print(current_set)
    best_set = current_set
    best_ever_acc = 0
    for i in range(np.size(data, 1)-1):
        print('On the ' + str(i), 'th level of the search tree')
        feature_to_remove = 9;
        worst_so_far_acc = 1;
        for k in range(np.size(data, 1)-1):
            if k in current_set:
                tempset = copy.deepcopy(current_set)
                tempset.append(k)
                print('--Considering removing the ', str(k+1) ,' feature')
                accuracy = cross_validation(data, tempset, k + 1, False) #pass in data, feature to remove, index of current feature
                if accuracy < worst_so_far_acc:
                    worst_so_far_acc = accuracy;
                    feature_to_remove = k;
        current_set.remove(feature_to_remove)
        if best_ever_acc < worst_so_far_acc:
            best_set = copy.deepcopy(current_set)
            best_ever_acc = worst_so_far_acc
        print('On level ', str(i+1),', I removed feature ', str(feature_to_remove+1), ' to current set, with accuracy: ' + str(worst_so_far_acc))
    return best_set, best_ever_acc

def feature_search_optimize(data):
    current_set = []
    best_set = []
    best_ever_acc = 0
    for i in range(np.size(data, 1)-1):
        print('On the ' + str(i), 'th level of the search tree')
        feature_to_add = 0;
        best_so_far_acc = 0.0;
        for k in range(np.size(data, 1)-1):
            if k not in current_set:
                tempset = copy.deepcopy(current_set)
                tempset.append(k)
                print('--Considering adding the ', str(k+1) ,' feature')
                accuracy = cross_validation(data, tempset, k + 1, True) #pass in data, feature to add, index of current feature
                
                if accuracy > best_so_far_acc:
                    best_so_far_acc = accuracy;
                    feature_to_add = k;    
        current_set.append(feature_to_add)
        print('On level ', str(i+1),', I added feature ', str(feature_to_add+1), ' to current set, with accuracy: ' + str(best_so_far_acc))
        if best_ever_acc < best_so_far_acc:
            best_set = copy.deepcopy(current_set)
            best_ever_acc = best_so_far_acc
        else:
            return best_set, best_ever_acc
    return best_set, best_ever_acc

def main():
#    data = np.loadtxt('CS170_LARGEtestdata__103.txt') #load relevant data.
    data = np.loadtxt('CS170_SMALLtestdata__15.txt')
    print('Welcome to Kau Chung\'s Feature Search Program!')
    print('Please enter 1, 2, or 3 for forward, backwards, or special search.')
    searchnum = int(input())

    start = time.time()
    if searchnum == 1:
        forward_result = feature_search_forward(data)
        copy_array = forward_result[0]
        for i in range(len(copy_array)):
            copy_array[i] = copy_array[i] + 1
        print('The best features are: ' + str(copy_array) + ' with accuracy ' + str(forward_result[1]))
    elif searchnum == 2:
        backward_result = feature_search_backward(data)
        copy_array = backward_result[0]
        for i in range(len(copy_array)):
            copy_array[i] = copy_array[i] + 1
        print('The best features are: ' + str(copy_array) + ' with accuracy ' + str(backward_result[1]))
    elif searchnum == 3:
        optimize_result = feature_search_optimize(data)
        copy_array = optimize_result[0]
        for i in range(len(copy_array)):
            copy_array[i] = copy_array[i] + 1
        print('The best features are: ' + str(copy_array) + ' with accuracy ' + str(optimize_result[1]))
    else:
        print('Error, invalid value')
        main()
        return
    end = time.time()
    print("Time to finish: " + str(end - start) + ' seconds')

if __name__ == "__main__":
    main()    
