import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
 
"""
YOLO11 旋转目标检测OBB
1、ONNX模型推理、可视化
2、ONNX输出格式: x_center, y_center, width, height, class1_confidence, ..., classN_confidence, angle
3、支持不同尺寸图片输入、支持旋转NMS过滤重复框、支持ProbIoU旋转IOU计算
"""
 
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=False, scale_fill=False, scale_up=False, stride=32):
    """
    将图像调整为指定尺寸，同时保持长宽比，添加填充以适应目标输入形状。
    :param img: 输入图像
    :param new_shape: 目标尺寸
    :param color: 填充颜色
    :param auto: 是否自动调整填充为步幅的整数倍
    :param scale_fill: 是否强制缩放以完全填充目标尺寸
    :param scale_up: 是否允许放大图像
    :param stride: 步幅，用于自动调整填充
    :return: 调整后的图像、缩放比例、填充尺寸(dw, dh)
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
    if not scale_up:
        r = min(r, 1.0)
 
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
 
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
 
    dw /= 2  # 填充均分
    dh /= 2
 
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)
 
def load_model(weights):
    """
    加载ONNX模型并返回会话对象。
    :param weights: 模型权重文件路径
    :return: ONNX运行会话对象
    """
    session = ort.InferenceSession(weights, providers=['CPUExecutionProvider'])
    logging.info(f"模型加载成功: {weights}")#写进日志
    return session
 
def _get_covariance_matrix(obb):
    """
    计算旋转边界框的协方差矩阵。
    :param obb: 旋转边界框 (Oriented Bounding Box)，包含中心坐标、宽、高和旋转角度
    :return: 协方差矩阵的三个元素 a, b, c
    """
    widths = obb[..., 2] / 2
    heights = obb[..., 3] / 2
    angles = obb[..., 4]
 
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
 
    a = (widths * cos_angle)**2 + (heights * sin_angle)**2
    b = (widths * sin_angle)**2 + (heights * cos_angle)**2
    c = widths * cos_angle * heights * sin_angle
 
    return a, b, c
 
def batch_probiou(obb1, obb2, eps=1e-7):
    """
    计算旋转边界框之间的 ProbIoU。
    :param obb1: 第一个旋转边界框集合
    :param obb2: 第二个旋转边界框集合
    :param eps: 防止除零的极小值
    :return: 两个旋转边界框之间的 ProbIoU
    """
    x1, y1 = obb1[..., 0], obb1[..., 1]
    x2, y2 = obb2[..., 0], obb2[..., 1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
 
    t1 = ((a1[:, None] + a2) * (y1[:, None] - y2)**2 + (b1[:, None] + b2) * (x1[:, None] - x2)**2) / (
            (a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2)**2 + eps) * 0.25
    t2 = ((c1[:, None] + c2) * (x2 - x1[:, None]) * (y1[:, None] - y2)) / (
            (a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2)**2 + eps) * 0.5
    t3 = np.log(((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2)**2) /
                (4 * np.sqrt((a1 * b1 - c1**2)[:, None] * (a2 * b2 - c2**2)) + eps) + eps) * 0.5
 
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd
 
def rotated_nms_with_probiou(boxes, scores, iou_threshold=0.5):
    """
    使用 ProbIoU 执行旋转边界框的非极大值抑制（NMS）。
    :param boxes: 旋转边界框的集合
    :param scores: 每个边界框的置信度得分
    :param iou_threshold: IoU 阈值，用于确定是否抑制框
    :return: 保留的边界框索引列表
    """
    order = scores.argsort()[::-1]  # 根据置信度得分降序排序
    keep = []
 
    while len(order) > 0:
        i = order[0]
        keep.append(i)
 
        if len(order) == 1:
            break
 
        remaining_boxes = boxes[order[1:]]
        iou_values = batch_probiou(boxes[i:i+1], remaining_boxes).squeeze(0)
 
        mask = iou_values < iou_threshold  # 保留 IoU 小于阈值的框
        order = order[1:][mask]
 
    return keep
 
def run_inference(session, image_bytes, imgsz=(640, 640)):
    """
    对输入图像进行预处理，然后使用ONNX模型执行推理。
    :param session: ONNX运行会话对象
    :param image_bytes: 输入图像的字节数据
    :param imgsz: 模型输入的尺寸
    :return: 推理结果、缩放比例、填充尺寸
    """
    im0 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # 解码图像字节数据
    if im0 is None:
        raise ValueError("无法从image_bytes解码图像")
 
    img, ratio, (dw, dh) = letterbox(im0, new_shape=imgsz)  # 调整图像尺寸
    img = img.transpose((2, 0, 1))[::-1]  # 调整通道顺序
    img = np.ascontiguousarray(img)
    img = img[np.newaxis, ...].astype(np.float32) / 255.0  # 归一化处理
 
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img})  # 执行模型推理
 
    return result[0], ratio, (dw, dh)
 
def parse_onnx_output(output, ratio, dwdh, conf_threshold=0.5, iou_threshold=0.5):
    """
    解析ONNX模型的输出，提取旋转边界框坐标、置信度和类别信息，并应用旋转NMS。
    :param output: ONNX模型的输出，包含预测的边界框信息
    :param ratio: 缩放比例，用于将坐标还原到原始尺度
    :param dwdh: 填充的宽高，用于调整边界框的中心点坐标
    :param conf_threshold: 置信度阈值，过滤低于该阈值的检测框
    :param iou_threshold: IoU 阈值，用于旋转边界框的非极大值抑制（NMS）
    :return: 符合条件的旋转边界框的检测结果
    """
    boxes, scores, classes, detections = [], [], [], []
    num_detections = output.shape[2]  # 获取检测的边界框数量
    num_classes = output.shape[1] - 6  # 计算类别数量
 
    # 逐个解析每个检测结果
    for i in range(num_detections):
        detection = output[0, :, i]
        x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]  # 提取边界框的中心坐标和宽高
        angle = detection[-1]  # 提取旋转角度
 
        if num_classes > 0:
            class_confidences = detection[4:4 + num_classes]  # 获取类别置信度
            if class_confidences.size == 0:
                continue
            class_id = np.argmax(class_confidences)  # 获取置信度最高的类别索引
            confidence = class_confidences[class_id]  # 获取对应的置信度
        else:
            confidence = detection[4]  # 如果没有类别信息，直接使用置信度值
            class_id = 0  # 默认类别为 0
 
        if confidence > conf_threshold:  # 过滤掉低置信度的检测结果
            x_center = (x_center - dwdh[0]) / ratio[0]  # 还原中心点 x 坐标
            y_center = (y_center - dwdh[1]) / ratio[1]  # 还原中心点 y 坐标
            width /= ratio[0]  # 还原宽度
            height /= ratio[1]  # 还原高度
 
            boxes.append([x_center, y_center, width, height, angle])  # 将边界框信息加入列表
            scores.append(confidence)  # 将置信度加入列表
            classes.append(class_id)  # 将类别加入列表
 
    if not boxes:
        return []
 
    # 转换为 NumPy 数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
 
    # 应用旋转 NMS
    keep_indices = rotated_nms_with_probiou(boxes, scores, iou_threshold=iou_threshold)
 
    # 构建最终检测结果
    for idx in keep_indices:
        x_center, y_center, width, height, angle = boxes[idx]  # 获取保留的边界框信息
        confidence = scores[idx]  # 获取对应的置信度
        class_id = classes[idx]  # 获取类别
        obb_corners = calculate_obb_corners(x_center, y_center, width, height, angle)  # 计算旋转边界框的四个角点
 
        detections.append({
            "position": obb_corners,  # 旋转边界框的角点坐标
            "confidence": float(confidence),  # 置信度
            "class_id": int(class_id),  # 类别 ID
            "angle": float(angle)  # 旋转角度
        })
 
    return detections
 
def calculate_obb_corners(x_center, y_center, width, height, angle):
    """
    根据旋转角度计算旋转边界框的四个角点。
    :param x_center: 边界框中心的 x 坐标
    :param y_center: 边界框中心的 y 坐标
    :param width: 边界框的宽度
    :param height: 边界框的高度
    :param angle: 旋转角度
    :return: 旋转边界框的四个角点坐标
    """
    cos_angle = np.cos(angle)  # 计算旋转角度的余弦值
    sin_angle = np.sin(angle)  # 计算旋转角度的正弦值
    dx = width / 2  # 计算宽度的一半
    dy = height / 2  # 计算高度的一半
 
    # 计算旋转边界框的四个角点坐标
    corners = [
        (int(x_center + cos_angle * dx - sin_angle * dy), int(y_center + sin_angle * dx + cos_angle * dy)),
        (int(x_center - cos_angle * dx - sin_angle * dy), int(y_center - sin_angle * dx + cos_angle * dy)),
        (int(x_center - cos_angle * dx + sin_angle * dy), int(y_center - sin_angle * dx - cos_angle * dy)),
        (int(x_center + cos_angle * dx + sin_angle * dy), int(y_center + sin_angle * dx - cos_angle * dy)),
    ]
    return corners  # 返回角点坐标
 
def save_detections(image, detections, output_path):
    """
    在图像上绘制旋转边界框检测结果并保存。
    :param image: 原始图像
    :param detections: 检测结果列表
    :param output_path: 保存路径
    """
    for det in detections:
        corners = det['position']  # 获取旋转边界框的四个角点
        confidence = det['confidence']  # 获取置信度
        class_id = det['class_id']  # 获取类别ID
 
        # 绘制边界框的四条边
        for j in range(4):
            pt1 = corners[j]
            pt2 = corners[(j + 1) % 4]
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        
        # 在边界框上方显示类别和置信度
        cv2.putText(image, f'Class: {class_id}, Conf: {confidence:.2f}', 
                    (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
 
    cv2.imwrite(output_path, image)  # 保存绘制后的图像
 
def process_images_in_folder(folder_path, model_weights, output_folder, conf_threshold, iou_threshold, imgsz):
    """
    批量处理文件夹中的图像，执行推理、解析和可视化，保存结果。
    :param folder_path: 输入图像文件夹路径
    :param model_weights: ONNX模型权重文件路径
    :param output_folder: 输出结果文件夹路径
    :param conf_threshold: 置信度阈值
    :param iou_threshold: IoU 阈值，用于旋转NMS
    :param imgsz: 模型输入大小
    """
    session = load_model(weights=model_weights)  # 调用函数，加载ONNX模型
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果输出文件夹不存在，则创建
    #循环处理多张图片
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 处理图片文件
            image_path = os.path.join(folder_path, filename)
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
 
            raw_output, ratio, dwdh = run_inference(session=session, image_bytes=image_bytes, imgsz=imgsz)  # 执行推理,包含预处理和推理过程
            detections = parse_onnx_output(raw_output, ratio, dwdh, conf_threshold=conf_threshold, iou_threshold=iou_threshold)  # 解析输出
 
            im0 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # 解码图像
            output_path = os.path.join(output_folder, filename)
            save_detections(im0, detections, output_path)  # 保存检测结果
 
# 主函数：加载参数
if __name__ == "__main__":
    folder_path = r"C:\Users\CHENZUMING\Desktop\rjgf\image"  # 推理图像文件夹路径
    model_weights = r"C:\Users\CHENZUMING\Desktop\rjgf\yolo11_obb\yolo11n-obb.onnx"  # ONNX模型路径
    output_folder = r"C:\Users\CHENZUMING\Desktop\rjgf\image_out"  # 推理结果输出文件夹
    conf_threshold = 0.4  # 置信度阈值
    iou_threshold = 0.5  # IoU阈值，用于旋转NMS
    imgsz = (1024, 1024)  # 模型输入大小
 
    process_images_in_folder(folder_path, model_weights, output_folder, conf_threshold, iou_threshold, imgsz)  # 执行批量处理