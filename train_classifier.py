import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu đã được lưu trong data.pickle
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    print("Error: 'data.pickle' not found!")
    exit()

# Lấy dữ liệu và nhãn
data = data_dict['data']
labels = data_dict['labels']

# Kiểm tra chiều dài của từng phần tử trong data và loại bỏ các phần tử rỗng
data = [item for item in data if len(item) > 0]  # Loại bỏ phần tử rỗng

# Kiểm tra xem có phần tử nào rỗng không
if len(data) == 0:
    print("Error: No valid data available!")
    exit()

# Kiểm tra chiều dài của dữ liệu còn lại
max_len = max(len(item) for item in data)

# Padding các phần tử trong data để tất cả đều có chiều dài bằng max_len
# Kiểm tra nếu chiều dài của phần tử hợp lệ (không rỗng) trước khi padding
data = [item + [0] * (max_len - len(item)) if len(item) > 0 else item for item in data]
data = np.asarray(data)

# Chuyển đổi nhãn từ dạng chuỗi thành số (nếu cần thiết)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Chia dữ liệu thành hai phần: một phần huấn luyện và một phần kiểm tra (80% huấn luyện, 20% kiểm tra)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Khởi tạo mô hình Random Forest 
model = RandomForestClassifier()

# Huấn luyện mô hình 
model.fit(x_train, y_train)

# Dự đoán trên tập kiểm tra
y_predict = model.predict(x_test)

# Tính toán độ chính xác của mô hình
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Lưu mô hình đã huấn luyện vào tệp model.p
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)