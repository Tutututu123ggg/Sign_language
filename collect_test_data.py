import os
import cv2

DATA_DIR = './data_test'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 10

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, chr(65 + j))  # Tên thư mục từ A-Z
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(chr(65 + j))) 

    # Giai đoạn chờ sẵn sàng
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start collecting for {}'.format(chr(65 + j)), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giai đoạn thu thập dữ liệu thủ công
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, 'Class {} - Image {}/{}'.format(chr(65 + j), counter + 1, dataset_size),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press "C" to capture, "ESC" to skip', 
                    (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Lưu ảnh khi nhấn 'c'
            img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
            cv2.imwrite(img_path, frame)
            print(f'Captured image {counter + 1} for class {chr(65 + j)}')
            counter += 1
        elif key == 27:  # ESC
            print('Skipping class {}'.format(chr(65 + j)))
            break

    print('Finished collecting data for class {}'.format(chr(65 + j)))

cap.release()
cv2.destroyAllWindows()
