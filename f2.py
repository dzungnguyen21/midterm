import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_items(image_path, template_path, scale_range=(0.5, 1), scale_step=0.01):
    # Đọc ảnh chính và template
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    # Chuyển ảnh sang grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng Canny Edge Detection
    edges_image = cv2.Canny(gray_image, 50, 150)
    
    best_match = None
    best_val = -np.inf
    best_rect = None
    
    # Duyệt qua các tỷ lệ
    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
        resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        edges_template = cv2.Canny(resized_template, 50, 200)
        
        result = cv2.matchTemplate(edges_image, edges_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            h, w = resized_template.shape
            best_rect = (best_match, (best_match[0] + w, best_match[1] + h))
    
    # Vẽ hình chữ nhật lên vị trí tìm được tốt nhất
    if best_rect:
        cv2.rectangle(image, best_rect[0], best_rect[1], (0, 255, 0), 2)
    
    # Hiển thị kết quả
    cv2.imshow('Detected Items', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

find_items('Finding/image.jpg', 'Template/find/dau.jpg')
