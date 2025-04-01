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
    
    best_matches = []

    # Duyệt qua các tỷ lệ
    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
        resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        edges_template = cv2.Canny(resized_template, 50, 200)
        
        result = cv2.matchTemplate(edges_image, edges_template, cv2.TM_CCOEFF_NORMED)
        h, w = resized_template.shape
        
        # Lấy top 10 vị trí khớp nhất
        num_results = 10
        locs = np.dstack(np.unravel_index(np.argsort(result.ravel())[::-1], result.shape))[:num_results]
        
        for loc in locs[0]:
            best_matches.append((result[loc[0], loc[1]], (loc[1], loc[0]), (loc[1] + w, loc[0] + h)))
    
    # Sắp xếp lại danh sách dựa trên giá trị matching score
    best_matches.sort(reverse=True, key=lambda x: x[0])
    best_matches = best_matches[:10]  # Chỉ giữ lại top 10
    
    # Vẽ hình chữ nhật lên vị trí tìm được tốt nhất
    for match in best_matches:
        cv2.rectangle(image, match[1], match[2], (0, 255, 0), 2)
    
    # Hiển thị kết quả
    cv2.imshow('Detected Items', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

find_items('Counting/cat.jpg', 'Template/count/cat.jpg')
