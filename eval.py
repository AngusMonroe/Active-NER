import matplotlib.pyplot as plt
import os
import numpy as np


def draw_one_method(path):
    plt.clf()
    file_names = os.listdir(path)
    x = []
    for name in file_names:
        if os.path.isdir(path + '/' + name):
            x.append(int(name))
    y_list = [[], [], [], [], []]
    for file_name in file_names:
        # print(file_name)
        if not os.path.isdir(path + '/' + file_name):
            continue
        f = open(path + '/' + file_name + '/score.txt', 'r', encoding='utf8')
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            if i == 1:
                y_list[i - 1].append(float(line.split('  ')[-1][:-1]))
            else:
                y_list[i - 1].append(float(line.split('  ')[-2]))
        f.close()
    # print(y_list)

    items = path.split('/')
    title = items[2] + '-' + items[3] + '-' + items[-1]
    plt.title(title)
    plt.plot(x, y_list[0], label='ALL')
    plt.plot(x, y_list[1], label='LOC')
    plt.plot(x, y_list[2], label='MISC')
    plt.plot(x, y_list[3], label='ORG')
    plt.plot(x, y_list[4], label='PER')
    plt.legend()
    plt.xlabel('train sentence#')  # make axis labels
    plt.ylabel('F1')
    my_y_ticks = np.arange(44, 95, 3)
    plt.yticks(my_y_ticks)
    plt.savefig(path + '/' + title + '.png')
    print(title + ' saved!')


def draw_overall(path):
    plt.clf()
    method_names = os.listdir(path)
    for method in method_names:
        if method == 'full':
            f = open(path + '/full/score.txt', 'r', encoding='utf8')
            plt.hlines(float(f.readlines()[1].split('  ')[-1][:-1]), 0, 102093, linestyles="dashed", label='full data')
            continue
        file_names = os.listdir(path + '/' + method)
        x, y = [], []
        for file_name in file_names:
            if not os.path.isdir(path + '/' + method + '/' + file_name):
                continue
            x.append(int(file_name))
            # print(file_name)
            f = open(path + '/' + method + '/' + file_name + '/score.txt', 'r', encoding='utf8')
            y.append(float(f.readlines()[1].split('  ')[-1][:-1]))
            f.close()
        plt.plot(x, y, label=method)

    items = path.split('/')
    title = items[2] + '-' + items[3] + '-overall'
    plt.title(title)
    plt.legend()
    plt.xlabel('train sentence#')  # make axis labels
    plt.ylabel('F1')
    my_y_ticks = np.arange(62, 94, 2)
    plt.yticks(my_y_ticks)
    plt.savefig(path + '/' + title + '.png')
    print(title + ' saved!')


if __name__ == '__main__':
    draw_one_method('neural_ner/results/conll/CNN_BiLSTM_CRF/active_checkpoint/random')
    # draw_overall('neural_ner/results/conll/CNN_BiLSTM_CRF/active_checkpoint')
