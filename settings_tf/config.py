



# 数据集存放路径
IMAGE_PATH = "./data/images/"


# label存放路径
IMAGE_LAEBL_PATH = './data/label.txt'


# 声明图片长宽和图层
IMAGE_SIZE = (60, 120, 1)


# 输出层， 例如4个数字[4, 10]， 5个小写字符加数字[5, 10+26]
LABEL_SIZE = (4, 36)

# 训练集和测试集比例
train_and_val = 0.2

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']