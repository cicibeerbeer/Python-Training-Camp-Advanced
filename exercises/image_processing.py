# exercises/image_processing.py
"""
练习：图像基本处理

描述：
使用 OpenCV 实现基本的图像读取、灰度转换、高斯滤波和边缘检测。

请补全下面的函数 `image_processing_pipeline`。
"""
import cv2
import numpy as np

def image_processing_pipeline(image_path):
    """
    使用 OpenCV 读取图像，进行高斯滤波和边缘检测。
    参数:
        image_path: 图像文件的路径 (字符串).
    返回:
        edges: Canny 边缘检测的结果 (NumPy 数组, 灰度图像).
               如果读取图像失败, 返回 None.
    """
    # 请在此处编写代码
    # 提示：
    # 1. 使用 cv2.imread() 读取图像。
    # 2. 检查图像是否成功读取（img is None?）。
    # 3. 使用 cv2.cvtColor() 将图像转为灰度图 (cv2.COLOR_BGR2GRAY)。
    # 4. 使用 cv2.GaussianBlur() 进行高斯滤波。
    # 5. 使用 cv2.Canny() 进行边缘检测。
    # 6. 使用 try...except 包裹代码以处理可能的异常。
    try:
        # 1. 使用 cv2.imread() 读取图像
        img = cv2.imread(image_path)
        
        # 2. 检查图像是否成功读取
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 3. 使用 cv2.cvtColor() 将图像转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4. 使用 cv2.GaussianBlur() 进行高斯滤波
        # 使用 5x5 的核，sigmaX=1.0
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # 5. 使用 cv2.Canny() 进行边缘检测
        # 使用较通用的阈值：低阈值50，高阈值150
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
        
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return None
    pass 