import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100  

cap = cv2.VideoCapture(0)

# Lặp qua từng lớp (mỗi lớp là một chữ cái từ A-Z)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, chr(65 + j))  
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(chr(65 + j))) 

    done = False
    while not done:
        ret, frame = cap.read()

        cv2.putText(frame, 'Ready? Press "Q" to start collecting data for {}'.format(chr(65 + j)), 
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            done = True

    # Thu thập dữ liệu (ảnh) cho lớp hiện tại
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        cv2.putText(frame, 'Collecting image {}/{}'.format(counter + 1, dataset_size),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        
        counter += 1

    print('Finished collecting data for class {}'.format(chr(65 + j)))

cap.release()
cv2.destroyAllWindows()
