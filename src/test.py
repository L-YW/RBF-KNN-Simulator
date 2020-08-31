"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com
"""

import argparse
from collections import Counter
import math
import numpy as np
import random as rd
#import log
#import image_procs as img_p
import cm1k_emulator as cm1k
import neuron as nrn
import csv
import gui

# Assign these to the database directories on disk
mnist_data_dir = None
faces_data_dir = None
iris_data_dir = None
mushroom_data_dir = None

def learn_test(data, radio=0, nmk=0, maxif=0, minif=0, num_board=1, verbose=0):
    """
    Test a basic single CM1K chip network with a single context.
    """
    train = data
    nmNk = nmk*1024
    print('nmk, nmNk', nmk, nmNk)
    if verbose >= 1:
        print ("\n\n\n\n\ntest_minimal_01")

    network = cm1k.CM1KEmulator(network_size=num_board * 1024)
    assert(not network.euclidean_norm)

    network.write_maxif(maxif)
    network.write_minif(minif)
    read_neuron_count=[0]
    # Train network(RBF Learning)
    if radio==1:

        l=len(train)
        iteration=0
        ID=0
        for i in range (0,2):
            if ID != l:#&(read_neuron_count[iteration] <= nmNk) :
                for input in train:
                    input_comps = [int(x) for x in input]
                    context = input_comps[0]
                    cat = input_comps[1]
                    pattern = input_comps[2:]
                    network.learn(pattern, cat, context)
                read_neuron_count.append(network.read_ncount())
                ID, UNC_c, UNC_i, UNK,total_detail, cla_result, null = classify(train, network, radio=5, spinvalue=3, num_board=1, verbose=1)
                iteration+=1
                # print(network.read_ncount())

            else :
                break
            #assert(network.read_ncount() == 3)
        # print('iteration', iteration)
        # print('maxif', maxif)
        # print('minif', minif)
        #print('network.register_legend[NSR] : ',network.register_legend['NSR'])
    # Write all sampels
    elif radio == 9:
        iteration = 0
        for input in train:
            # if verbose >= 1:
            # print "================================================================================"
            input_comps = [int(x) for x in input]
            context = input_comps[0]
            cat = input_comps[1]
            pattern = input_comps[2:]
            network.learn_write_all(pattern, cat, context)
        # print(network.read_ncount())
    #elif radio==2: #Deep RBF
    return network, iteration, read_neuron_count

def classify(test, network, rbfknn=0, radio=0, spinvalue=0, convalue=0, num_board=1, verbose=1):
    """
    Test a basic single CM1K chip network with a single context.
    """
    value4 = spinvalue #value4 : K
    consensus=convalue #value5 : Min consensus of N
    if rbfknn==3:
        network.rbforknn=0
    elif rbfknn==4:
        network.rbforknn=1
    #print(rbfknn)
    #print(network.rbforknn)
    #print (network.neurons[0:11])
    # Train network
    patternID = []
    test_detail=[]
    total_detail = []
    i=1
    ID = 0
    UNC_c = 0
    UNC_i = 0
    UNK = 0
    incorrect=0
    cla_result=[]
    classification_result=0
    cat_acurracy=[]
    j=1
    for input in test:
        input_comps = [int(x) for x in input]
        test_context = input_comps[0]
        test_cat = input_comps[1]
        test_pattern = input_comps[2:]
        test_detail.append([input_comps[0],input_comps[1]])
        patternID.append(i)
        i+=1
        id_, unc, firing_neurons = network.broadcast(input_=test_pattern, new_gcr=test_context)
        #print(firing_neurons)
        temp=0
        temp2=0
        temp3=0
        temp4=0
        temp5=0
        cat=[]
        for firing_neuron in firing_neurons:
            cat.append(firing_neuron.cat)
        if len(cat)>=value4:
            iter=value4
        else :
            iter=len(cat)
        #print '=========================================='
        #print('network.read_cat() : ', cat)
        #print('iter : ', iter)

        #########Category Out#########
        #Best match
        if radio==5:
            best_neuron = firing_neurons[-1] if firing_neurons else None
            if best_neuron != None:
                classification_result = cat[0]
            else :
                classification_result = None
            #print('classification_result : ', classification_result)

        #Dominant
        elif radio==6:
            if iter>=1:
                for i in range(iter):
                    if cat[i]==1:
                        temp+=1
                    elif cat[i]==2:
                        temp2+=1
                    elif cat[i]==3:
                        temp3+=1
                    elif cat[i]==4:
                        temp4+=1
                    # elif cat[i]==5:
                    #     temp5+=1
                list=[temp, temp2, temp3, temp4]
                value=max(list)

                if value==temp:
                    classification_result = 1
                elif value == temp2:
                    classification_result = 2
                elif value == temp3:
                    classification_result = 3
                elif value == temp4:
                    classification_result = 4
                # elif value==temp5:
                #     classification_result = 5
            else: #iter==0
                classification_result = None
            #print('classification_result : ', classification_result)
        #Unanimity
        elif radio==7:
            if iter>=1:
                for i in range(iter):
                    if cat[i] == test_cat:
                        temp += 1
                    else:
                        temp -= 1
                if temp == (iter):
                    classification_result = test_cat
                else:
                    classification_result = None
            else: #iter==0
                classification_result = None
            #print('temp : ', temp)
            #print('classification_result : ', classification_result)
        #Min consensus of N(value5)
        elif radio==8:
            for i in range(iter):
                if cat[i] == 1:
                    temp += 1
                elif cat[i] == 2:
                    temp2 += 1
                elif cat[i] == 3:
                    temp3 += 1
                elif cat[i] == 4:
                    temp4 += 1
                # elif cat[i]==5:
                #     temp5+=1
            list = [temp, temp2, temp3, temp4]
            value = max(list)
            if value >= consensus:
                if value == temp:
                    classification_result = 1
                elif value == temp2:
                    classification_result = 2
                elif value == temp3:
                    classification_result = 3
                elif value == temp4:
                    classification_result = 4
                # elif value==temp5:
                #     classification_result=5
            else:
                classification_result=None
            #print('classification_result : ', classification_result)
        cla_result.append(classification_result)

        #Accuracy
        #print('test_cat == classification_result ?', test_cat, classification_result)
        l=len(cat)
        if l==1 or l==0:
            if classification_result == test_cat :
                ID +=  1
                cat_acurracy.append([test_cat, 'ID'])
            elif classification_result == None:
                UNK += 1
                cat_acurracy.append([test_cat, 'UNK'])
            elif classification_result != test_cat :
                incorrect +=1
        elif l==2:
            if cat[0] == cat[1] :
                if classification_result == test_cat:
                    ID += 1
                    cat_acurracy.append([test_cat, 'ID'])
            else:
                if classification_result == None:
                    UNK += 1
                    cat_acurracy.append([test_cat, 'UNK'])
                elif classification_result != test_cat:
                    UNC_i += 1
                    cat_acurracy.append([test_cat, 'UNC_i'])
                elif classification_result == test_cat:
                    UNC_c += 1
                    cat_acurracy.append([test_cat, 'UNC_c'])
        elif l>=3:
            if cat[0] == cat[1] and cat[0] == cat[2]:
                if classification_result == test_cat:
                    ID += 1
                    cat_acurracy.append([test_cat, 'ID'])
            else:
                if classification_result == None:
                    UNK += 1
                    cat_acurracy.append([test_cat, 'UNK'])
                elif classification_result != test_cat:
                    UNC_i += 1
                    cat_acurracy.append([test_cat, 'UNC_i'])
                elif classification_result == test_cat:
                    UNC_c += 1
                    cat_acurracy.append([test_cat, 'UNC_c'])

        #sum=ID+UNC_i+UNC_c+UNK
        #print("ID, UNC_c, UNC_i, UNK", j, ID, UNC_c, UNC_i, UNK, sum)
        #print(l)
        j+=1
        #table data
        #print(cat_acurracy)
        detail = []
        for firing_neuron in firing_neurons:
            detail.append([firing_neuron.cat,firing_neuron.dist, firing_neuron.id_ ])
        #print(detail)
        patternID = np.reshape(patternID, (1, -1))
        test_detail = np.reshape(test_detail, (1, -1))
        detail = np.reshape(detail, (1, -1))
        if classification_result != None:
            temp_detail = np.hstack((patternID, test_detail, detail))
        else:
            temp_detail = np.hstack((patternID, test_detail))
        test_detail= []
        patternID = []
        total_detail.append(temp_detail)
        #print(temp_detail)

    #print (ID/32.16)
    #print(UNC_c/32.16)
    #print(UNC_i/32.16)
    #print(UNK/32.16)
    return ID, UNC_c, UNC_i, UNK, total_detail, cla_result, cat_acurracy