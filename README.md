# ML-Object-Detection
RCNN/Faster RCNN/
How to understand a picture? There are three main levels depending on the subsequent task.
One is Classification, which is the process of structuring an image into a certain category of information and describing the picture with a pre-determined category (string) or instance ID. This task is the simplest and most basic image understanding task, and it is also the first task for deep learning models to make breakthroughs and achieve large-scale applications. Among them, ImageNet is the most authoritative review set, and the annual ILSVRC has spawned a large number of excellent deep network structures that provide the basis for other tasks. In the application domain, recognition of faces, scenes, etc. can be classified as classification tasks.
The second is Detection. Classification tasks are concerned with the whole, giving a description of the content of the whole picture, while Detection focuses on a specific object target, requiring both the category information and location information of this target. In contrast to classification, detection gives an understanding of the foreground and background of the image, we need to separate the target of interest from the background and determine the description (category and location) of this target, thus, the output of the detection model is a list, and each item of the list uses a data set to give the category and location of the detected target (often represented by the coordinates of a rectangular detection box).
Third, Segmentation. Segmentation consists of semantic segmentation, which is an extension of the former background separation that requires separating image parts with different semantics, and instance segmentation, which is an extension of the detection task that requires describing the contours of the target (which is more fine-grained than the detection frame). Segmentation is a pixel-level description of an image that assigns meaning to each pixel category (instance), and is suitable for scenarios with high comprehension requirements, such as the segmentation of roads and non-roads in unmanned driving.
In this session, we will get into the learning and practice of Object Detection, from the cornerstone RCNN and its variants to first-order models such as SSD and Yolo to the industry's cutting-edge algorithms.

RCNN
As RCNN is relatively old and basic, there are not many implementations in the industry, so you need to implement RCNN for Object Detection on your own after understanding the algorithm. https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with- keras-tensorflow-and-deep-learning/

As a test, we use Coco dataset 2014: https://cocodataset.org/#home to select bicycle, motorcycle, car, traffic light, stop sign, bus, truck, train, person classification and train the model and then perform network image testing.

Faster RCNN
Faster RCNN is a variant of RCNN, based on RCNN has been significantly modified and solved many known problems. Faster RCNN has been a popular choice for 2-stage detection algorithms in the current industry.
tfhub already has pre-trained models with ResNet Inception classifiers trained on OpenImagesNet, even out-of-the-box.
For the tf implementation of the paper we can refer to: https://github.com/endernewton/tf-faster-rcnn
One problem that needs to be addressed in this task is the method of evaluating the performance of Object Detection models.

# ML-Object-Detection
RCNN/Faster RCNN/
如何理解一张图片？根据后续任务的需要，有三个主要的层次。
一是分类（Classification），即是将图像结构化为某一类别的信息，用事先确定好的类别(string)或实例ID来描述图片。这一任务是最简单、最基础的图像理解任务，也是深度学习模型最先取得突破和实现大规模应用的任务。其中，ImageNet是最权威的评测集，每年的ILSVRC催生了大量的优秀深度网络结构，为其他任务提供了基础。在应用领域，人脸、场景的识别等都可以归为分类任务。
二是检测（Detection）。分类任务关心整体，给出的是整张图片的内容描述，而检测则关注特定的物体目标，要求同时获得这一目标的类别信息和位置信息。相比分类，检测给出的是对图片前景和背景的理解，我们需要从背景中分离出感兴趣的目标，并确定这一目标的描述（类别和位置），因而，检测模型的输出是一个列表，列表的每一项使用一个数据组给出检出目标的类别和位置（常用矩形检测框的坐标表示）。
三是分割（Segmentation）。分割包括语义分割（semantic segmentation）和实例分割（instance segmentation），前者是对前背景分离的拓展，要求分离开具有不同语义的图像部分，而后者是检测任务的拓展，要求描述出目标的轮廓（相比检测框更为精细）。分割是对图像的像素级描述，它赋予每个像素类别（实例）意义，适用于理解要求较高的场景，如无人驾驶中对道路和非道路的分割。
这一环节我们将入手进行Object Detection的学习与实践，从奠基的RCNN及其变体到SSD、Yolo等一阶模型逐步深入到行业前沿算法。

RCNN
由于RCNN比较旧且基础，行业目前没有太多的实现案例，因而需要在理解算法后自行实现RCNN的Object Detection。参考：https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/

作为测试，我们使用Coco dataset 2014：https://cocodataset.org/#home，选择bicycle, motorcycle, car, traffic light, stop sign, bus, truck, train, person分类，训练出模型后进行网络图像测试。

Faster RCNN
Faster RCNN是RCNN的一个变体，在RCNN的基础上做了大幅修改并且解决了很多已知的问题。当前行业流行的2 stage detection算法中，Faster RCNN一直是热门的选择。
tfhub已经有搭配ResNet Inception分类器在OpenImagesNet训练好的预训练模型，甚至做到开箱即用。
论文的tf实现我们可以参考：https://github.com/endernewton/tf-faster-rcnn
在这一任务需要解决一个问题，即Object Detection模型性能的评估方法。
