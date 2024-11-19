# detect-and-track-objects

Tổng quan
Dự án này triển khai hệ thống phát hiện và theo dõi thời gian thực sử dụng YOLO , Kalman Filters và MediaPipe để phát hiện, theo dõi và nhận dạng khuôn mặt đối tượng. Hệ thống nhận dạng con người, theo dõi chuyển động của họ và thực hiện nhận dạng khuôn mặt bằng cách so sánh các nhúng khuôn mặt đã phát hiện với dữ liệu tham chiếu được tải trước. Các khuôn mặt chưa được nhận dạng được lưu để phân tích thêm.

# Đặc trưng
Phát hiện đối tượng dựa trên YOLO : Phát hiện con người theo thời gian thực bằng cách sử dụng mô hình YOLO được đào tạo trước.
Theo dõi bộ lọc Kalman : Làm mượt chuyển động của vật thể và cung cấp khả năng theo dõi vật thể bằng cách dự đoán.
Nhận dạng khuôn mặt :
Trích xuất nhúng khuôn mặt 3D bằng MediaPipe FaceMesh.
So sánh khuôn mặt được phát hiện với tập dữ liệu tham chiếu dựa trên độ tương đồng cosin.
Lưu khuôn mặt chưa khớp : Lưu trữ cục bộ các khuôn mặt chưa khớp để xử lý thêm.
Hình ảnh trực quan thời gian thực :
Vẽ các hộp giới hạn và ID cho con người được phát hiện.
Hiển thị trạng thái "Trùng khớp" hoặc "Chưa trùng khớp" cho khuôn mặt.
