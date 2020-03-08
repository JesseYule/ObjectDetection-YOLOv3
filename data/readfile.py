datanum = 100

# 取10%的数据作为测试集
testnum = round(datanum * 0.1)
testidx = round(datanum/testnum)

label = open("label.txt", 'a')
train = open("train.txt", 'a')
test = open("test.txt", 'a')
label.seek(0)
label.truncate()
train.seek(0)
train.truncate()
test.seek(0)
test.truncate()

for i in range(datanum):

    num = 1 + i
    filename = 'smoke-' + str(num)
    file = 'bbox/' + filename + '.txt'

    f = open(file)
    content1 = f.readlines()
    axis = content1[1]
    result = filename + '.JPEG ' + axis
    result = '../data/images/' + result.strip('\n') + ' 0'
    label.write(result)
    label.write("\n")

    if i % testidx == 0:
        test.write(result)
        test.write("\n")
    else:
        train.write(result)
        train.write("\n")
