![图片](https://images-cdn.shimo.im/RyZeWpX5kCASZ2pp/image.png!thumbnail)
速写训练集
![图片](https://images-cdn.shimo.im/Mg9DuBIwmrAxHB5P/image.png!thumbnail)
速写增强后训练集
![图片](https://images-cdn.shimo.im/qMmY80aoShsH0ziO/image.png!thumbnail)
                                                               色彩训练集
![图片](https://images-cdn.shimo.im/McRwXgcGS7Q3rXlm/image.png!thumbnail)
                                                               色彩测试集

用增强数据做了一个预测统计：准确率（5分差之内）：69.57%
![图片](https://images-cdn.shimo.im/CemOYp9Pauw2q1qZ/image.png!thumbnail)
                                                          测试集标签
![图片](https://images-cdn.shimo.im/04dPskmyrMI0a75F/image.png!thumbnail)
                                                        测试集prediction





---
近期工作总结：
1、数据集整理：
*        切割不规则图片数据：由于数据集是手机拍照，有些美术图片的只占据了整个图片的一部分，并且不规整。所以使用了Recursive  CNN对图片进行切割，然后使用opencv提取。
*        图片方向对齐：数据集中图片的方向不是按照统一的方向，这里主要采用的方法是手动把图片的的方向归类为0、90、180、270四个类，然后根据每个类对图片做旋转操作
*        图片的分数提取：使用目标检测网络，检测图片的分数的位置，并把检测到的图片分数框截取处理，然后使用OCR把图片分数提取出来。
*        把提取出的结果按照分数放置在同一个文件夹中，挑选出提取错误的分数，并整理

目前数据集共有6W张图片



---
## 
## 任务 to do
建议：清洗数据、相关 batch 实验可以同时进行，互补耽误（下周重点在加粗 + 下划线）

从诸多因素来测算不同数据集、类别数、模型、损失函数对结果影响，即相当于 batch experiment 探索该模型对最后结果的影响
* **修改类别（10-20 或者更多）**
* **修改模型（ResNet，VGG）**
* **修改损失函数**

     网络训练方面，通过对预测阶段的数据处理方法的更改（之前是直接对预测图片进行随机切割，更改后对图片从四个角和中心各裁剪，然后求平均）在测试集上准确率有2个点的提升

数据清洗：调整不同数据集对实验结果影响，期待达到两个效果：
* **对百度网盘数据集进行清洗：分数、方向、裁剪**

**   **  目前完成了对网盘数据集的裁剪和方向的清洗，分数现在做到了把图片对应的分数给截取出来，分数的OCR识别未完成

* 增强训练数据集：改进图像增强方法，简单裁剪容易过拟合，更好的扭曲变形可能。是图像增强的一个有效手段：[https://github.com/mdbloice/Augmentor](https://github.com/mdbloice/Augmentor)
* 训练一个多数据集融合的 basic model，基础模型，在此模型基础上：
* 预测新考试，得到考试分布基本上合理、准确，正确率在 80%
* 如遇见新考试，可以 fine-turn 一些老师打过的样本，再预测效果依旧可以


---
* 整理数据集，按照之前给的格式，对本地数据整理
目前按照第一次考试的数据集已经整理完成，第二次考试的数据集由于不规整，目前还没有整理D:\works\data\exam_data\exam_01(第一次考试)
* 给出一份机器与专家评分融合数据
![图片](https://images-cdn.shimo.im/x11msz1FiFcqjvPb/image.png!thumbnail)
* 统计正负 6 分误差分布是多少，看看是不是更合理？满足内容
正负 6 分误差：准确率在76.7%
![图片](https://images-cdn.shimo.im/SD3AV2QNVXAVk4zX/image.png!thumbnail)

* 修改 Loss Function，但是我感觉会差不多，仔细看定义是一回事，去掉 abs
* 选择合适的类别，在去掉 abs 情况下，10、15、20、25 等试验

         #exam_03 20分类对比图
   ![图片](https://images-cdn.shimo.im/6puxWfxlxdQVCG3I/image.png!thumbnail)


任务认知
* 这是一份 score 任务，和 Google 对图像打分是类似逻辑和场景
* 分数跨度为 0-120 或者 0-60 分，要从数据分布来选择粒度粗细，避免过于集中

从上述数据可得，明显数据不均衡，建议如下操作：
* 数据处理：直接 reset 成最原始状态，重新跑一次，思考下为什么有问题
* 数据均衡：将数据全部均衡为 600 / 1000，以保证模型充分收集各类样本
* 标签粗细：先取最细粒度作为 score 标签，再逐步加粗，看分类效果，确保不出现全是 40 分情况
* 单独模型：每类数据，单独训练一个模型，确保一个模型学习到类似东西
* 可视结果：给出模型相关正确率，以及画出 predict 分数分布

参考文献
* 数据均衡，[https://www.leiphone.com/news/201807/n29cU8N7Jk5w0K7E.html](https://www.leiphone.com/news/201807/n29cU8N7Jk5w0K7E.html)
* 图像裁剪，[https://github.com/Khurramjaved96/Recursive-CNNs](https://github.com/Khurramjaved96/Recursive-CNNs)

