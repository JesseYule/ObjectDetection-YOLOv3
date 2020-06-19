import xml.dom.minidom
from PIL import Image

datanum = 1083
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

for n in range(1, 1083):

    doc = 'Annotations/img'+str(n)+'.xml'
    dom = xml.dom.minidom.parse(doc)

    xmin = dom.getElementsByTagName('xmin')
    ymin = dom.getElementsByTagName('ymin')
    xmax = dom.getElementsByTagName('xmax')
    ymax = dom.getElementsByTagName('ymax')

    target_num = len(xmin)

    result = []

    img_name = '../data/voc_data/images/img'+str(n)+'.jpg'

    img_path = 'images/img'+str(n)+'.jpg'
    img = Image.open(img_path)
    size = img.size
    img_x = size[0]
    img_y = size[1]

    result.append(img_name)

    for i in range(target_num):
        if i > 3:
            break

        xmin_value = xmin[i].firstChild.data
        ymin_value = ymin[i].firstChild.data
        xmax_value = xmax[i].firstChild.data
        ymax_value = ymax[i].firstChild.data

        # 部分数据标注的时候超出了原始尺寸，这里需要处理一下
        if float(xmax_value) > img_x:
            xmax_value = img_x - 1
            print("change x")

        if float(ymax_value) > img_y:
            ymax_value = img_y - 1
            print("change y")

        if float(xmin_value) > img_x:
            xmin_value = img_x - 1
            print("change x")
            print(img_name, "x")

        if float(ymin_value) > img_y:
            ymin_value = img_y - 1
            print("change y")
            print(img_name, "y")


        result.append(xmin_value)
        result.append(ymin_value)
        result.append(xmax_value)
        result.append(ymax_value)
        result.append(0)  # 最后添加的0是类别，这里只有一个目标类别就用0表示,多目标时根据实际情况添加

    output = str(result)
    output = output.replace(',', '')
    output = output.replace('[', '')
    output = output.replace(']', '')
    output = output.replace("'", '')

    if i % testidx == 0:
        test.write(output)
        test.write("\n")
    else:
        train.write(output)
        train.write("\n")





