import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Tải mô hình đã huấn luyện
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

# Tải dữ liệu test
with open("data_test.pickle", "rb") as f:
    data_dict = pickle.load(f)

X_test = np.asarray(data_dict["data"])
y_test = np.asarray(data_dict["labels"])

# Kiểm tra và điều chỉnh số chiều nếu cần
expected_features = model.n_features_in_  # Số đặc trưng mà model yêu cầu
if X_test.shape[1] < expected_features:
    print(f"[!] Dữ liệu test có {X_test.shape[1]} đặc trưng, nhưng mô hình yêu cầu {expected_features}.")
    print("    -> Đang tự động padding bằng 0 cho đủ chiều...")
    pad_width = expected_features - X_test.shape[1]
    X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')
elif X_test.shape[1] > expected_features:
    print(f"[!] Dữ liệu test có {X_test.shape[1]} đặc trưng, nhưng mô hình yêu cầu {expected_features}.")
    print("    -> Cắt bớt cho khớp với mô hình...")
    X_test = X_test[:, :expected_features]

# Dự đoán
y_pred = model.predict(X_test)

# Nếu là nhãn số (0–25), chuyển thành A–Z
if isinstance(y_pred[0], (int, np.integer)):
    y_pred = [chr(label + 65) for label in y_pred]

# Tính độ chính xác
acc = accuracy_score(y_test, y_pred)
print("Độ chính xác trên tập test: {:.2f}%".format(acc * 100))

# In báo cáo chi tiết
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, zero_division=0))

# Vẽ ma trận nhầm lẫn
labels = sorted(list(set(y_test)))
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn trên tập test")
plt.tight_layout()
plt.show()
