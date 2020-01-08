import matplotlib.pyplot as plt
import os


def draw(path):
    plt.clf()
    file_names = os.listdir(path)
    x = [int(name) for name in file_names]
    y_list = [[], [], [], [], []]
    for file_name in file_names:
        # print(file_name)
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
    plt.savefig(path + '/' + title + '.png')
    print(title + ' saved!')


if __name__ == '__main__':
    draw('neural_ner/results/conll/CNN_BiLSTM_CRF/active_checkpoint/random')
