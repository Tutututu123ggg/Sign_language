import os  
import pickle  
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data_test'  # <--- thư mục test

data = [] 
labels = [] 

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  
        x_ = []  
        y_ = [] 

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue  # bỏ qua ảnh lỗi

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)  
            labels.append(dir_)  

# Lưu thành file test riêng
with open('data_test.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✓ Đã tạo tập dữ liệu test: data_test.pickle ({} mẫu)".format(len(data)))
