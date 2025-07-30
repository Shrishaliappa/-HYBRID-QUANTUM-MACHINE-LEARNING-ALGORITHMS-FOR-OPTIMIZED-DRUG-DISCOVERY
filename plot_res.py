import pandas as pd
import matplotlib.pyplot as plt
import os
from save_load import *
def bar_plot(label, data1, data2, metric):
    # create data
    df = pd.DataFrame([data1, data2], columns=label)
    df1 = pd.DataFrame()
    df1['Training percentage(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)

    # Create plot with larger figure size
    plt.figure(figsize=(10, 6))  # <-- Increased plot size

    # plot grouped bar chart
    df.plot(x='Training percentage(%)',
            kind='bar',
            stacked=False,
            fontsize=16,
            ax=plt.gca())  # Use current axis for better control

    plt.ylabel(metric, fontsize=16)
    plt.xlabel("Training percentage (%)", fontsize=16)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='center', fontsize=16)

    if not os.path.exists('./baseline_Model_Result'):
        os.makedirs('./baseline_Model_Result')

    plt.savefig('./baseline_Model_Result/' + metric + '.png', dpi=1100)
    plt.show(block=False)



def polt_res():

    Metrices=load('baseline_Existing_Model')
    mthod = ['RF', 'XGBoost', 'GCN', 'ChemBERTa', 'Proposed']
    metrices_plot = ['Accuracy',' Precision', 'Specificity', 'Sensitivity', 'NPV', 'F-measure', 'MCC', 'FPR', 'FNR']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, Metrices[0][i, :], Metrices[1][i, :], metrices_plot[i])

    for i in range(2):
        # Table
        print('Metrices-Dataset- ' + str([i]))
        tab = pd.DataFrame(Metrices[i], index=metrices_plot, columns=mthod)
        print(tab)
        excel_file_path = './baseline_Model_Result/table_dataset' + str(i + 1) + '.xlsx'
        tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column


polt_res()


def line_plot(label, data1, data2, metric):
    # Create a DataFrame
    df = pd.DataFrame({
        'Training percentage(%)': [70, 80]
    })
    for i, lbl in enumerate(label):
        df[lbl + ' (70%)'] = [data1[i], None]
        df[lbl + ' (80%)'] = [None, data2[i]]

    # Melt the DataFrame for line plotting
    df_melted = pd.melt(df, id_vars=['Training percentage(%)'], var_name='Metric', value_name=metric)

    # Plot
    plt.figure(figsize=(10, 6))
    for lbl in label:
        plt.plot([70, 80], [data1[label.index(lbl)], data2[label.index(lbl)]],
                 marker='o', linewidth=2, label=lbl)

    plt.xlabel("Training percentage (%)", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.xticks([70, 80], fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f"{metric} vs Training Percentage", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Save if directory does not exist
    if not os.path.exists('./Existing_Model_Result'):
        os.makedirs('./Existing_Model_Result')

    plt.savefig(f'./Existing_Model_Result/{metric}_line.png', dpi=1100)
    plt.show(block=False)


def polt_res():
    Metrices = load('Metrices_Existing_Model')
    mthod = ['SVM [22]', 'CycleGAN [24]', 'CNN-Siam[26]', '3D CNN[27]', 'Proposed']
    metrices_plot = ['Accuracy', ' Precision', 'Specificity', 'Sensitivity', 'NPV', 'F-measure', 'MCC', 'FPR', 'FNR']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, Metrices[0][i, :], Metrices[1][i, :], metrices_plot[i])

    for i in range(2):
        # Table
        print('Metrices-Dataset- ' + str([i]))
        tab = pd.DataFrame(Metrices[i], index=metrices_plot, columns=mthod)
        print(tab)
        excel_file_path = './Existing_Model_Result/table_dataset' + str(i + 1) + '.xlsx'
        tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column


polt_res()