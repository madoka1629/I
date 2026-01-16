from ctypes import sizeof
import re
import csv
import numpy as np
import jieba
jieba.set_dictionary('dict.txt')  # 內建為中文，使用繁體字典


def text_process(tweet):  # 清洗空格
    after_wash = re.sub(
        " ", "", tweet)
    return after_wash


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding="utf-8").readlines()]
    return stopwords


# 設定 stopwords字典，用於告訴 jieba 哪些是停頓詞
stopwords = stopwordslist('C:\Topic\stop.txt')

# 計算正面貼文中的詞頻
with open('pos.csv', newline='', encoding="utf-8-sig") as csvfile:
    i = 0
    pos_words = []  # 正面詞
    arr_pos = []
    # 讀取 CSV 檔案內容
    data = csv.reader(csvfile)
    # 以迴圈輸出每一列
    for row in data:  # row 代表每則 tweets
        temp = str(row)
        arr_pos.append(temp)
        output = text_process(temp)
        sentence = jieba.cut(output, cut_all=False)  # 斷詞
        # for tk in sentence:
        # print(tk)
        for word in sentence:
            if word not in stopwords:  # 停用詞及標點符號
                pos_words.append(word)

        freq_pos = {}  # 建立頻率字典
        for word in pos_words:

            if (word, 1) not in freq_pos:
                freq_pos[(word, 1)] = 1
            else:
                freq_pos[(word, 1)] = freq_pos[(word, 1)]+1
    # print('\n正面字典\n')
    # print(freq_pos.items())


with open('neq.csv', newline='', encoding="utf-8-sig") as csvfile:
    # 讀取 CSV 檔案內容
    data = csv.reader(csvfile)
    neg_words = []  # 負面詞
    arr_neg = []
    # 以迴圈輸出每一列
    for row in data:  # 每一筆tweets
        temp = str(row)
        arr_neg.append(temp)
        output = text_process(temp)  # 清洗完
        sentence = jieba.cut(output, cut_all=False)  # 分詞

        # for tk in sentence:
        # print(tk)
        for word in sentence:  # 合併
            if word not in stopwords:  # 停用詞及標點符號
                neg_words.append(word)

        freq_neg = {}
        for word in neg_words:
            if (word, 0) not in freq_neg:
                freq_neg[(word, 0)] = 1
            else:
                freq_neg[(word, 0)] = freq_neg[(word, 0)]+1

    freqs_dict = dict(freq_pos)
    freqs_dict.update(freq_neg)


def features_extraction(tweet, freqs_dict):  # 傳入貼文和字典
    B = []
    word = text_process(tweet)  # 清洗完後
    x = np.zeros((1, 3))
    x[0, 0] = 1
    sentence = jieba.cut(word, cut_all=False)  # 分詞
    for w in sentence:  # 合併
        B.append(w)
    l = len(B)
    for i in range(l):
        try:
            x[0, 1] += freqs_dict[(B[i], 1)]
        except:
            x[0, 1] += 0
        try:
            x[0, 2] += freqs_dict[(B[i], 0)]
        except:
            x[0, 2] += 0
    assert(x.shape == (1, 3))
    return x


def sigmoid(x):  # sigmoid函數
    h = 1/(1+np.exp(-x))
    return h


def gradientDescent_algo(x, y, theta, alpha, num_iters):  # 梯度下降
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1/m*(np.dot(y.T, np.log(h))+np.dot((1-y).T, np.log(1-h)))
        theta = theta-(alpha/m)*np.dot(x.T, h-y)
    J = float(J)
    return J, theta


n_pos = 25000
n_neg = 25000

# 正面資料
train_pos = arr_pos[:int(0.8*n_pos)]   # 0 ~ 19999
test_pos = arr_pos[int(0.8*n_pos):]   # 20000 ~ 24999

# 負面資料
train_neg = arr_neg[:int(0.8*n_neg)]   # 0 ~ 19999
test_neg = arr_neg[int(0.8*n_neg):]   # 20000 ~ 24999

# 合併資料集
train_x = train_pos + train_neg
test_x = test_pos + test_neg


train_y = np.append(np.ones((len(train_pos), 1)),
                    np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)),
                   np.zeros((len(test_neg), 1)), axis=0)

# 計算權重
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = features_extraction(train_x[i], freqs_dict)
Y = train_y

J, theta = gradientDescent_algo(X, Y, np.zeros((3, 1)), 1e-9, 1500)


def predict(tweet, freqs_dict, theta):
    x = features_extraction(tweet, freqs_dict)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


def test_accuracy(test_x, test_y, freqs_dict, theta):
    y_hat = []
    for tweet in test_x:

        y_pred = predict(tweet, freqs_dict, theta)

        if y_pred > 0.5:

            y_hat.append(1)
        else:

            y_hat.append(0)
    m = len(y_hat)
    y_hat = np.array(y_hat)
    y_hat = y_hat.reshape(m)
    test_y = test_y.reshape(m)

    c = y_hat == test_y
    j = 0
    count = 0
    for i in c:
        if i == True:
            j = j+1
            count = count+1
        else:
            print(test_x[count])
            count = count+1
    accuracy = j/m
    return accuracy


accuracy = test_accuracy(test_x, test_y, freqs_dict, theta)
print("正確率", accuracy)
