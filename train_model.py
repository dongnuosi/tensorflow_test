"""
基于vgg16框架的的图像识别
该文件主要包括数据集加载、图像数据预处理、构造数据生成器、构建模型、加载权重、训练模型、评估模型等流程。
"""

import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator # jia zai zhengsshi shuju ji ,ji tupian ku
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import  GlobalAveragePooling2D  #zuiida zhi chi hua , ji zai sige zhi zhong qu zuida zhi ##It doesn't matter that there are somes redlinies follows import likes this sentence,be carefull:just from ... import ...,,,not import...
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint#ji lu xunlian zhong zui hao yici de quan zhong
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

t1 = time.time()

"""
功能：设置参数
"""
num_classes  = 4#4 lei
batch_size   = 64
epochs       = 5#p yang ben shu ,5 lun ,di er lun he di san lun shi zhunquedu da dao le 98,99 ,dan shi 5 ci hou bing mei you lixiang,,hou mian juanji ceng mei yong ,yong le vgg xun lian hao de ,50 ci shi xun lian hou wending xiaoguo hao
iterations   = 50


"""
功能：设置GPU
"""
from tensorflow.keras import backend as K
if('tensorflow' == K.backend()):  #fangzhi zhan yong suo you de xian cun
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


################################## 构造数据生成器阶段 ###################################
"""
功能：数据集信息
"""
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'


"""
功能：设置数据生成器对象参数
"""
# 通过数据生成器，达到获取数据集的效果
datagen = ImageDataGenerator(horizontal_flip=True,  # ziji sou xia mei ge canshu shi ganshenme de
                             rescale=1. / 255,
                             rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2)


"""
功能：数据生成器
"""
# flow_from_directory(directory)：从文件夹中获取数据流，所以将训练集、验证集的地址作为参数输入即可。
train_datagenerator = datagen.flow_from_directory(  #cong wen jian jia zhong huo qu shuju liu
        train_dir,
        target_size=(224, 224), # 输出图像大小 ,ben lai bu shi zhe daxiao ,yao tiaozheng
        batch_size=batch_size,
        class_mode='categorical'
    )


val_datagenerator = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )


test_datagenerator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )


##################################构建模型、添加全连接层、编译模型阶段 ###################################
"""
功能：构建模型
"""
base_model = VGG16(include_top=False,weights='imagenet') # 获取的VGG16模型(zui da de shu ju ji)，只作为特征提取器，不包含全连接层


"""
功能：添加全连接层
"""
x = base_model.output
x = GlobalAveragePooling2D()(x) #quan ju junzhi chi hua
x = Dense(4096, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
predictions = Dense( num_classes, activation='softmax')(x) # 输出层，一共训练四种类别, di er ceng juan ji ,shu chu si wei fen lei ,
model = Model(inputs=base_model.input, outputs=predictions)


"""
功能：设置TensorBoard,以便可视化某些参数的变化，如loss、accuracy等
"""
import time
tb_cb = TensorBoard(log_dir=f"log/{time.strftime('%Y-%m-%d-%H-%M-%S')}", histogram_freq=0)
best_model = ModelCheckpoint('trained_model/best_train_model.h5', # baocun xun lian guocheng zhong zui hao de quanzhong
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only = False)
cbks = [best_model,tb_cb]


# ################################## 训练模型###################################
"""
功能：冻结vgg16全部层
"""
# 如图片少，数据量小，则冻结所有层，只训练全连接层；如数据量大，则可冻结部分层，训练部分卷积层和全连接层
for layer in base_model.layers:
    layer.trainable = False


"""
功能：编译模型
"""
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

"""
功能：训练模型
"""
model.fit_generator(train_datagenerator,
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=val_datagenerator)


###################################评估模型####################################
"""
功能：评估模型
"""
score = model.evaluate_generator(test_datagenerator)
# 输出结果
print(f'Test score:{score[0]}')
print(f'Accuracy:{score[1]*100}')
print('完成！')

#####################################保存模型#######################################
"""
功能：保存模型
"""
if score[1] > 0.8:
    model.save('trained_model/save_train_model.h5')

t2 = time.time()
print(f'time:{t2 - t1}')