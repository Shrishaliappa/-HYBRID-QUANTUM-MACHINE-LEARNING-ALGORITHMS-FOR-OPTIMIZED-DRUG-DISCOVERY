import matplotlib
from save_load import load, save
matplotlib.use('TkAgg', force=True)
from Data_gen import *
from Quantum_Boosting_Models import *
from Comparision_model import *
from plot_res import *
from BCOA import *
def full_analysis():
    Data_gen()
    X_train_70 = load('X_train_70')
    X_test_30 = load('X_test_30')
    y_train_70 = load('y_train_70')
    y_test_30 = load('y_test_30')
    X_train_80 = load('X_train_80')
    X_test_20 = load('X_test_20')
    y_train_80= load('y_train_80')
    y_test_20= load('y_test_20')




    #Training percentage(70% and 30%)
    #PROPOSED
    met,latency_ms= proposed(X_train_70, X_test_30,y_train_70,y_test_30)
    save('proposed_met_70',met)

    #SVM [22]
    met,latency_ms = svm_ga(X_train_70, X_test_30,y_train_70,y_test_30)
    save('CNN_TSODE_met_70', met)

    #CycleGAN [24]
    cm,latency_ms =CycleGAN(X_train_70, X_test_30,y_train_70,y_test_30)
    save('CycleGAN_met_70',cm)

    #CNN-Siam[26]
    cm,latency_ms =cnn_model(X_train_70, X_test_30,y_train_70,y_test_30)
    save('CNN-Siam_met_70', cm)

    #3D CNN[27]
    cm,latency_ms =D3cnn(X_train_70, X_test_30,y_train_70,y_test_30)
    save('SVM_GA_met_70', cm)


    # Training percentage(80% and 20%)
    # PROPOSED
    met,latency_ms = proposed(X_train_80, X_test_20, y_train_80, y_test_20)
    save('proposed_met_80', met)

    #SVM [22]
    met,latency_ms = svm_ga(X_train_80, X_test_20, y_train_80, y_test_20)
    save('svm_met_80', met)

    # CycleGAN [24]
    cm,latency_ms= CycleGAN(X_train_80, X_test_20, y_train_80, y_test_20)
    save('CycleGAN_met_80', cm)

    #CNN-Siam[26]
    cm,latency_ms = cnn_model(X_train_80, X_test_20, y_train_80, y_test_20)
    save('CNN-Siam_met_80', cm)

    #3D CNN[27]
    cm,latency_ms = D3cnn(X_train_70, X_test_30, y_train_70, y_test_30)
    save('d3cnn_met_80', cm)



a =0
if a == 1:
    full_analysis()

polt_res()
