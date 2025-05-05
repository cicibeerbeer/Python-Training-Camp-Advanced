"""
练习：轮廓检测
描述：
使用 OpenCV 检测图像中的轮廓并将其绘制出来。
请补全下面的函数 `contour_detection`。
"""
import cv2
import numpy as np

def contour_detection(image_path):
    """
    使用 OpenCV 检测图像中的轮廓
    参数:
        image_path: 图像路径
    返回:
        tuple: (绘制轮廓的图像, 轮廓列表) 或 (None, None) 失败时
    """
    try:
        # 1. 使用 cv2.imread() 读取图像
        img = cv2.imread(image_path)
        
        # 2. 检查图像是否成功读取
        if img is None:
            return None, None
        
        # 3. 使用 cv2.cvtColor() 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4. 使用 cv2.threshold() 进行二值化处理
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 5. 使用 cv2.findContours() 检测轮廓
        # OpenCV 版本差异处理
        contours_result = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理不同版本 OpenCV 的返回值差异
        if len(contours_result) == 3:
            _, contours, hierarchy = contours_result
        else:
            contours, hierarchy = contours_result
            
        # 确保 contours 是一个列表类型
        contours = list(contours)
        
        # 6. 创建图像副本用于绘制
        result = img.copy()
        
        # 7. 使用 cv2.drawContours() 在副本上绘制轮廓
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # 8. 返回绘制后的图像和轮廓列表
        return result, contours
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None