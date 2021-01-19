# 莫娜图片分离器
将图片内容分为：画面里有莫娜和画面里没有莫娜两类
详情可以移步[我的博客](http://www.erlnesa.com/2021/01/莫娜图片分离器/)

## 最初的需求

​		莫娜实在是太铯了（虽然到现在也没有抽到XD）

​		不过没抽到并不影响我去网上找**莫娜**的同人图来，嗯，收藏。但这个时候我就发现了一个问题。拿pixiv来举例：

​		如果想要搜索**莫娜**的图片，正常来说都会以**莫娜**作为**tag**进行检索。但如果真的这样检索，会发现一个问题：

​		返回的搜索结果里，虽然图片里都是莫娜没错，但**图片的数量非常少**。这对于pixiv这个超大型的网站和全球人民的XP系统来说，是非常异常的现象。

​		为啥会发生这样的情况？

​		经过一段时间的观察我发现，有些作者只给自己的作品打上了**原神**这个tag，而没有**莫娜**，或者是使用了除了中文外的其他语言拼写的**莫娜**作为tag，比如说日文的**モナ**。

​		更有些情况下，无论是作品的标题还是tag里都没有出现莫娜，但图片里的内容确是莫娜。

​		**这样的情况并不少见**，因此现在可以得出一个结论：

​		并不是全球人民的XP系统出现了问题，而是我们的搜索方法出现了亿点点问题。

​		那既然这样，完全可以搜索一个范围更大的tag，比如说原神，或者是Genshin，然后在返回的搜索结果里，手动寻找内容是莫娜的图片。这应该也是较为常见的方法了。

​		优点是精确度极高，缺点就是费时费力。

​		大胆一点，如果我想一打开电脑里的某个隐藏文件夹，点几下鼠标，就能看到尽可能多的莫娜图片呢？

​		于是本文就这个“朴实的”需求展开研究。


## 需求分析

​		我们的需求只有一个：

​		一打开电脑里的某个隐藏文件夹，点几下鼠标，就能看到尽可能多的莫娜图片。

​		**如何获取到尽可能多的莫娜图片？**

​		可以根据上文提到过的方法，以原神或者Genshin这样的tag来进行大范围的搜索。之后再对搜索到的图片进行分离。筛选出画面里有莫娜的图片，然后保存下来。

​		那么现在大问题变成了两个小问题：

​		首先是：**怎么快速大量获得原神或者Genshin的tag的图片？**

​		其次是：**怎么从上一步获取到的图片中，检索到莫娜相关的图片？**

​		第一个问题可以利用**爬虫**来大量获取图片，这将是后续主要研究的问题，本文先就第二个问题进行研究。

​		上学期刚结课的神经网络还热乎，因此马上准备用**卷积神经网络**——图像**二分类**。

​		所有的原神图片，需要分为两类，分别是：

​		**“画面里有莫娜”，“画面里没有莫娜”**

​		找了一圈资料后选择了目前网上资料相当多的**Tensorflow**和**python**来解决这个问题。

​		利用Tensorflow可以快速搭建出一个可以运行起来的神经网络模型，同时Tensorflow也支持利用GPU来进行计算。

## 环境搭建

​		首先是在**win10**的环境下安装**python3.8**，**tensorflow-gpu-2.4**，**cudn11.1**，**cudnn8.0**

### 安装Python3.8

​		Python3.8的安装非常简单，在官网下载安装包，一路next，**注意中间要勾选上path**。

​		安装完毕后重启电脑。之后小娜搜索cmd，打开命令提示符，输入：

```
python
```


​		回车。如果像这样显示出了版本号则安装成功，现在可以关闭命令提示符了。

​		**如果提示python不是可执行程序，则需要重新安装python，并注意是不是在安装过程中忘记勾选了path。**

​		接下来需要为python进行**换源**操作。

​		因为各种原因，如果python使用默认的官方下载源去下载各种包会非常，非常的慢，所以这里需要换成国内的高速镜像源。

​		重新打开命令提示符，输入：

```
pip install pqi
```

​		等待安装完毕后输入：

```
pqi ls
```

​		这时会列出目前可用的告诉镜像源：


​		这里**切换到清华大学的镜像**。

```
pqi use tuna
```

​		完毕后关闭命令提示行。

​		到此python环境的部署已经完成了。

### 安装tensorflow-gpu-2.4

​		小娜搜cmd，打开命令提示符，输入：

```
pip install tensorflow-gpu
```

​		整个下载安装的过程是全自动的，如果你发现下载过程非常慢，则需要回到上一步，利用pqi进行换源操作。

​		安装完毕后，命令行输入：

```
python
```

​		进入python解释器界面后，输入：

```python
import tensorflow as tf
```

​		如果有提示信息输出（因为还没有安装cudn，所以也可能会输出例如未找到某些dll的错误信息），则代表tensorflow-gpu已经被安装


​		**重启计算机**，现在已经完成了tensorflow-gpu的安装。接下来还需要安装cuda和cudnn来保证tensorflow-gpu的正常运行。

### 安装cuda11.1

​		[下载cuda](https://developer.nvidia.com/cuda-11.1.0-download-archive)


​		我强烈建议**下载local安装包**，如果游览器下载速度慢，你应该是需要一些特殊操作。

​		下载完毕之后进行安装，一路next。需要注意的是，**安装选项不要选择精简**，改成自定义以后，选择全部安装。

​		安装完毕以后能在开始菜单里发现这样的东西：


​		小娜搜索环境变量，可以找到一个叫做**CUDA_PATH_V版本号**的环境变量，例如这里是CUDA_PATH_V11_1


​		如果没有出现这些，则需要重新安装cuda。

### 安装cudnn8.0

​		[下载cudnn](https://developer.nvidia.com/rdp/cudnn-archive)

​		选择与cuda版本对应的cudnn下载。下载时需要注册并登录英伟达的开发者账号。

​		同样，如果下载速度太慢，你就需要考虑一些特殊操作了。

​		下载完毕后会得到一个这样的安装包：


​		打开这个压缩包，同时再打开刚才cuda的安装路径，默认是

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1
```

​		将压缩包里对应文件夹里的东西，拖到cuda的目录里。比如说压缩包里的bin文件夹里的东西，复制到cuda的bin文件夹里。


​		复制完毕后**重启计算机**。

​		命令行输入python，再次输入：

```python
import tensorflow as tf
```

​		如果没有显示未找到某些dll库，则安装成功。

​		如果再后续的使用中，提示某些dll库未找到，则可根据提示的信息来重新安装cuda和cudnn。

​		例如提示未找到**cuda_某某某_11.dll**

​		你就需要重新安装cuda11，cudnn也类似。提示的缺失文件里一般都有需要安装的版本号。

### 安装Pycharm

​		上面这些全部完成后。使用**学信网**的学籍证明申请一个Pycharm的**教育许可**。

​		或者也可以买，或者也可以，嗯你懂的。我因为怕麻烦所以是直接申请的教育许可，从申请到审批花了大概一周的时间。

​		到此为止已经跨出了第一步，环境搭建。


#### 设置Python环境

​		由于Pycharm自带一个python解释器，而自带的解释器是没有安装过tensorflow的，所以需要给Pycharm**切换解释器**。

​		在**设置**——**Python解释器**中将解释器从默认的内置解释器，调整到之前安装的Python3.8


​		确认、重新启动Pycharm。到此为止可以新建一个Python项目，试试能否成功导入tensorflow库。

## 收集训练数据集

​		因为目标是将图片**二分类**，所以训练样本也要分成两类。


​		如图，这里是分为了**mona**，**other**这两类。

​		之后就是找各种图片来放入这两个文件夹，**mona**文件夹里放入画面里有莫娜的图片，**other**文件夹里放入任何画面里没有莫娜的图片。

​		因为这个过程是纯手动的，所以最终我收集到的图片数量比较少：


​		最终手动得到了两个文件夹合计300张图片。

​		新建一个叫mona的文件夹，然后将刚才的两个文件夹放入其中，**压缩成zip**。


​		注意压缩以后不要把原文件夹删了。

​		另外，**手动收集到的图片格式有些是png，有些是jpg**。

​		对于**png**格式的图片来说，他比**jpg**格式的图片额外多出了一个**透明度图层**（阿尔法图层），这在后续的训练中意义并不大，所以需要**将png格式的图片全部变成jpg格式**。

​		这里使用python来完成这个任务：

```python
from PIL import Image
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import random
import os

# 导入数据
data_root_orig = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
data_root = pathlib.Path(data_root_orig)
# 解析
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
# 图片总数
image_count = len(all_image_paths)

for img_index in range(len(all_image_paths)):
    img_is_png = all_image_paths[img_index].endswith(".png")
    if img_is_png :
        #print(all_image_paths[img_index].replace(".png",".jpg"))
        image = Image.open(all_image_paths[img_index])
        image_rgb = image.convert('RGB')
        image_rgb.save(all_image_paths[img_index].replace(".png",".jpg"))
        # 移除
        os.remove(all_image_paths[img_index])
        #print(all_image_paths[img_index])

print("Done")
```

​		其中：

```python
tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
```

​		zip的路径换成刚才压缩包的路径，mona换成存放mona、other这两个文件夹的文件夹名。

​		运行后程序会把所有png图片替换为jpg图片。

​		如果import语句报错，提示缺少相关的包，则需要在Pycharm的设置——Python解释器：


​		点左下角的加号进行安装：


## 导入数据集

​		后续的步骤参考自：[Tensorflow官网教程](https://tensorflow.google.cn/tutorials/images/classification#import_tensorflow_and_other_libraries)

​		不管怎么说，现在得到了一个只有jpg图片的数据集。新建一个python文件，将它们导入进Tensorflow：

```python
import numpy as np
import os
# 隐藏tensorflow的输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import PIL
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 导入训练数据集
data_dir = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
```

​		运行，控制台此时会输出两个文件夹内的图片总数：我的情况是300。


​		到此已经成功的把搜集到的数据导入进了tensorflow中。

## 格式化图片

​		对于刚才手动找到的训练样本来说，**每张图片的分辨率是不一致的**，举个例子：


​		这两张图片的分辨率分别是1000×1630和2125×1500

​		这个情况对于后续的训练是非常不利的，所以需要将所有的图片转换为长度一致，宽度一致的形状。这里就定为300×300。

```python
# 格式化训练数据
batch_size = 32
img_height = 300
img_width = 300
```

## 拆分训练数据集

​		接下来需要将训练用的数据集进行拆分，一部分数据用来训练神经网络，一部分则会用来检测训练结果。

```python
# 拆分出训练数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# 拆分出验证数据集
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
# 您将通过传递这些数据集来训练使用这些数据集的模型model.fit。
# 如果愿意，还可以手动遍历数据集并检索成批图像：
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
# 暂时不知道是干啥的，照抄官网教程
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

​		这里我们设置了比例为0.2，也就是说**80%的图片会被用来训练**，另外**20%则会用来检验训练结果**。

​		到此再运行，程序会给出拆分的结果：


## 数据标准化和数据扩充

​		因为**手动收集到的样本实在是太少了**，这对于训练神经网络来说是**非常不利**的，因此这里引入**数据标准化**和**数据扩充**来弥补这个缺陷：

```python
# 标准化数据
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 数据扩充
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
```

​		到此为止，导入的数据已经准备好开始训练了。

## 创建卷积神经网络模型

### 层结构与Dropout

​		该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有128个单元。

​		激活函数为**relu**。

​		模型输出类为两类。

​		同时为了避免出现过拟合：


​		这里还引入了Dropout。

```python
# 创建模型
# 该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有128个单元。
# 可以通过relu激活功能激活。尚未针对高精度调整此模型，本教程的目的是展示一种标准方法。
num_classes = 2
# Dropout
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# 对于本教程，选择optimizers.Adam优化器和losses.SparseCategoricalCrossentropy损失函数。
# 要查看每个训练时期的训练和验证准确性，请传递metrics参数。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 查看网络的所有层
model.summary()
```

​		此时程序会给出可以训练的节点数量等参数：


## 开始训练

​		设置**迭代次数为15次**，开始训练上面建立的神经网络模型：

```python
#开始训练
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```


## 可视化训练结果

​		接下来可以绘制一个图像，来表示该模型训练的过程和最终的准确程度：

```python
# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

​		运行可以得到这样的结果：


## 验证

​		到此为止已经完成了神经网络的训练，现在可以利用训练好的模型来对已有的图片进行分类了。

​		待识别的图片路径设定为了**D:/huangdoufen.jpg**

```python
mona_test_dir = str("D:/huangdoufen.jpg")
mona_path = tf.keras.utils.get_file('mona_test', origin = mona_test_dir)

img = keras.preprocessing.image.load_img(
    mona_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "这张图片最接近分类 {} ，置信度为 {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 300])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)



plt.imshow(load_and_preprocess_image(mona_test_dir))
plt.grid(False)
plt.xlabel(
    "percent confidence: {:.2f} %"
    .format(100 * np.max(score))
)
plt.title(class_names[np.argmax(score)])
plt.show()
print()
```

​		结果如下：


​		也可以改变图片路径，测试一下另外的图片：


​		到此为止，建立的模型已经可以在一定程度上识别出莫娜了。接下来就是将它嵌入到图片爬虫中去：

​		当爬虫爬取到一张新的图片，就用它来识别看看这张图片里有没有莫娜，有的话就保存下来。

​		这个会在后续的文章中继续更新。

## 现在的问题

​		模型的识别**成功率低**，并且如果莫娜摆出奇怪的姿势，模型也无法正常识别。目前推测的原因是训练样本数量太少，神经网络的模型也有待优化。

​		**目前还不知道应该如何修改。**

​		自己尝试过增加迭代次数：


​		但看起来好像没有什么用。

​		也尝试过增加图片格式化中的分辨率，使图片保留更多的信息

​		但显存顶不住：


​		目前准备尝试调整卷积核的大小，神经网络的结构再尝试一下。

## 源代码

​		**训练集来自Pixiv**：[Pixiv](https://www.pixiv.net/)

​		**代码参考Tensorflow官网教程修改**：[Tensorflow官网教程](https://tensorflow.google.cn/tutorials/images/classification#import_tensorflow_and_other_libraries)

```python
import matplotlib.pyplot as plt
import numpy as np
import os
# 隐藏tensorflow的输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import PIL
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 导入训练数据集
data_dir = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# 格式化训练数据
batch_size = 32
img_height = 300
img_width = 300
# 拆分出训练数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# 拆分出验证数据集
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
# 您将通过传递这些数据集来训练使用这些数据集的模型model.fit。
# 如果愿意，还可以手动遍历数据集并检索成批图像：
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
# 暂时不知道是干啥的，照抄官网教程
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# 标准化数据
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 数据扩充
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# 创建模型
# 该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有128个单元。
# 可以通过relu激活功能激活。尚未针对高精度调整此模型，本教程的目的是展示一种标准方法。
num_classes = 2
# Dropout
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# 对于本教程，选择optimizers.Adam优化器和losses.SparseCategoricalCrossentropy损失函数。
# 要查看每个训练时期的训练和验证准确性，请传递metrics参数。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 查看网络的所有层
model.summary()
#开始训练
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

mona_test_dir = str("D:/baigong.jpg")
mona_path = tf.keras.utils.get_file('mona_test', origin = mona_test_dir)

img = keras.preprocessing.image.load_img(
    mona_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "这张图片最接近分类 {} ，置信度为 {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [200, 200])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)



plt.imshow(load_and_preprocess_image(mona_test_dir))
plt.grid(False)
plt.xlabel(
    "percent confidence: {:.2f} %"
    .format(100 * np.max(score))
)
plt.title(class_names[np.argmax(score)])
plt.show()
print()
```


