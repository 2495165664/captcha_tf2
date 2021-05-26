import sys


def run(location, amount, display):
    """
    进度条展示
    :param location:当前进度
    :param amount: 总进度条数
    :return:
    """
    percent = 1.0 * location / amount * 100
    sys.stdout.write(f"\r{display}  进度: %d / %d" % (percent, 100) + "%")
    if location == amount:
        sys.stdout.write("\n")
    sys.stdout.flush()
