import pickle
import cv2
import mediapipe as mp
import numpy as np

# Đặt tên file video ,để là None để dùng webcam
video_path = "WIN_20250623_21_12_53_Pro.mp4"  #file video đặt cùng thư mục file này nên chỉ cần tên

# Tải mô hình từ file pickle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Khởi tạo webcam hoặc video file
cap = cv2.VideoCapture(video_path if video_path else 0)

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Không thể mở video hoặc webcam.")
    exit()

# Khởi tạo các module của MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Tạo đối tượng Hands để phát hiện các dấu vân tay trên ảnh
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionary ánh xạ số từ 0-25 sang các chữ cái A-Z
labels_dict = {i: chr(i + 65) for i in range(26)}

# Vòng lặp chính để thu thập và xử lý ảnh
while True:
    data_aux = []  # Lưu trữ dữ liệu đặc trưng
    x_ = []  # Lưu trữ các tọa độ x
    y_ = []  # Lưu trữ các tọa độ y

    # Đọc ảnh từ webcam hoặc video
    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape  # Kích thước ảnh

    # Chuyển ảnh từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Sử dụng MediaPipe để phát hiện landmarks trên tay
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:  # Nếu phát hiện được tay
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các landmarks trên tay
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Chuẩn hóa các tọa độ và lưu vào data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Tọa độ x chuẩn hóa
                data_aux.append(y - min(y_))  # Tọa độ y chuẩn hóa

        # Đảm bảo rằng data_aux có 42 đặc trưng, nếu không, padding với 0
        max_len = 42  # Mô hình yêu cầu 42 đặc trưng
        if len(data_aux) > max_len:
            data_aux = data_aux[:max_len]  # Cắt bớt nếu quá dài
        elif len(data_aux) < max_len:
            data_aux = data_aux + [0] * (max_len - len(data_aux))  # Padding nếu quá ngắn

        # Dự đoán chữ cái từ mô hình
        prediction = model.predict([np.asarray(data_aux)])

        # Ánh xạ kết quả dự đoán từ chỉ số thành chữ cái
        predicted_character = labels_dict[int(prediction[0])]

        # Tính toán tọa độ để vẽ hình chữ nhật quanh tay
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Vẽ hình chữ nhật quanh tay và hiển thị chữ cái dự đoán
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Hiển thị kết quả
    cv2.imshow('frame', frame)
    print("Bấm \"Q\" ở terminal để dừng");
    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
