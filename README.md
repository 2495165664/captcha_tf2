# captcha_tf2 
验证码识别 深度学习 tensorflow 神经网络<br> 
使用卷积神经网络，对字符，数字类型验证码进行识别，`tensorflow`使用`2.0`以上<br>
<br>
目前项目还在更新中，欢迎提出issue和PR, 希望和你一起共同完善项目。

# 实例demo
## 训练过程
> 目前训练val提升可以，loss下降稳定
>> ![](./data/used_images/train_1.png)<br>
> demo图片<br>
>>  ![效果](./data/used_images/1.jpg) 
>>  ![效果](./data/used_images/2.jpg) 
 
 
# 目录
1. 项目结构
> 1.1 文件目录
>> |序号 | 文件| 说明|
>> | ----- | ----- | -----|
>> | 1 | model/ | 模型权重文件| 
>>| 2 | network/ | 神经网络|
>> | 3 | settings_tf | 项目配置文件|
>> | 4 | tools/ | 工具文件 |
>> | 5 | data/ | 数据文件|

> 1.2 主要文件
>> | 序号 | 文件 | 说明|
>> |------|------|-----|
>> | 1 | train.py | 训练程序 |
>> | 2 | detect.py | 测试程序 |
>> | 3 | make_data.py | 训练集合成程序|
>> | 4 | create_image.py | 数据集生产脚本|


2. 项目介绍
3. 使用