# bigming
this a good obb
代码中，主要依赖opencv、onnxruntime、numpy这三个库,是不需要依赖ultralytics的
对于opencv，主要用途是图像预处理、后处理和可视化，
对于onnxruntime，主要用途是加载ONNX模型并执行推理
对于numpy，用于数据格式转换、数学计算和数组操作

整个obb推理过程
​​预处理阶段​​：
OpenCV解码图像 → NumPy处理数组 → OpenCV调整尺寸 → NumPy归一化并转置通道顺序。
​​推理阶段​​：
ONNX Runtime加载模型 → 输入NumPy数组格式的图像 → 输出预测结果（NumPy数组）。
​​后处理阶段​​：
NumPy解析边界框坐标 → 应用旋转NMS（基于NumPy的矩阵运算） → OpenCV绘制旋转框并保存。
