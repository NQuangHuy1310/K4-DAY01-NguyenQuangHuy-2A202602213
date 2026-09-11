# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** Chưa xác định CPU/GPU từ artifact đã lưu.

**Python / PyTorch / Ultralytics:** Python `3.xx` và PyTorch; Ultralytics `8.4.145`.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không ghi nhận thay đổi trong các artifact đầu ra.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1: `class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`.
- Record này xếp lớp `cab` cho toàn bộ cảnh `traffic`, dựa trên các đặc trưng chung của ảnh như đường phố và nhiều phương tiện; nó không mô tả riêng một chiếc xe cụ thể.
- Class list do taxonomy ImageNet-1K dùng để huấn luyện checkpoint định nghĩa; model chỉ dự đoán trong các lớp đã học, không tự tạo ra lớp mới.
- Cần giữ cả ID, tên lớp và taxonomy vì ID chỉ có ý nghĩa trong đúng bảng lớp, tên lớp giúp con người đọc, còn taxonomy cho biết bảng ánh xạ đang được sử dụng.
- Với ảnh có nhiều chủ thể, guideline cần quy định cách chọn nhãn cấp ảnh: ưu tiên chủ thể chính, hoạt động/cảnh tổng thể hay một tiêu chí khác; đồng thời cần nêu cách xử lý trường hợp không có chủ thể nổi bật hoặc có nhiều lớp cùng phù hợp.
- Model score chỉ thể hiện mức ưu tiên/tự tin tương đối của model trong các lớp ImageNet-1K. Nó chưa được con người kiểm tra theo guideline nên không phải ground truth và không tự chứng minh prediction là đúng.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record: `class_name="person"`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68`.
- Box bao quanh người đứng ở nửa bên phải ảnh `kitchen`: góc trên-trái tại khoảng `(385.33, 69.24)` và góc dưới-phải tại khoảng `(498.92, 348.92)`, theo đơn vị pixel và gốc tọa độ ở góc trên-trái ảnh.
- Với sample `kitchen`, output ở threshold `0.35` có 11 prediction. Nếu chỉ giữ các record có score từ `0.60` trở lên thì còn 6 prediction.
- Threshold cao hơn làm giảm số prediction và khối lượng reviewer phải xem, nhưng cũng làm giảm độ bao phủ và tăng nguy cơ bỏ sót object. Threshold thấp hơn giữ được nhiều ứng viên hơn nhưng reviewer phải kiểm tra thêm prediction yếu và false positive.
- Quy tắc box chặt đề xuất: box phải là hình chữ nhật nhỏ nhất bao hết phần nhìn thấy của object, bám sát các điểm ngoài cùng và không cố ý chứa thêm nền hoặc object lân cận.
- Guideline cần quyết định mức độ nhìn thấy tối thiểu để gán nhãn, box chỉ bao phần nhìn thấy hay ước lượng toàn bộ object, và cách xử lý object chạm/cắt mép ảnh. Nếu không đủ căn cứ hoặc có nhiều cách hiểu hợp lý, annotator phải đánh dấu và escalation cho reviewer/lead thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record: `instance_id="kitchen-001"`, `class_name="person"`, `score=0.899318`, `polygon_point_count=348`; các điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
- Polygon mô tả đường biên và vùng pixel của người chi tiết hơn box chữ nhật, nhờ đó tách được hình dáng cơ thể khỏi phần nền nằm bên trong box.
- `instance_id` phân biệt từng object riêng trong output, kể cả khi nhiều object có cùng class. Nó không phải `class_id`, COCO image ID hay tracking ID dùng để theo dõi object qua nhiều khung hình.
- Quy tắc biên mask đề xuất: polygon phải bám sát phần object thực sự nhìn thấy, không lấy nền; giữ các phần thuộc object có thể nhận biết rõ và không tự vẽ xuyên qua vùng bị che để đoán hình dạng không quan sát được.
- Guideline cần quy định cách xử lý biên tóc/quần áo, lỗ hoặc khoảng trống, vật tiếp xúc nhau, vùng nhòe và phần bị che khuất. Khi không xác định chắc pixel thuộc instance nào, annotator cần đánh dấu vùng mơ hồ và escalation cho reviewer/lead.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth                       | Lỗi hoặc điểm mơ hồ quan sát được                                                                                   | Annotator làm gì?                                                                             | Reviewer xem gì?                                                                                      |
| --------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn lớp cho toàn ảnh theo taxonomy đã quy định | Ảnh `traffic` có nhiều phương tiện nhưng model xếp `cab` hạng 1, nên chủ thể/nhãn cấp ảnh có thể mơ hồ              | Chọn nhãn theo quy tắc về chủ thể chính hoặc loại cảnh; không chép top-1 của model thành nhãn | Kiểm tra nhãn có đúng taxonomy, đúng phạm vi toàn ảnh và nhất quán với các ảnh nhiều chủ thể tương tự |
| Phát hiện vật thể     | Một class và một box `xyxy` cho mỗi object          | Vật thể nhỏ, bị che hoặc cắt mép dễ bị bỏ sót; prediction người ở mép trái ảnh `kitchen` cần được kiểm tra bằng mắt | Gán đủ object thuộc phạm vi và vẽ box sát phần guideline yêu cầu, kể cả object model bỏ sót   | Kiểm tra sai lớp, box thừa nền, box thiếu object, duplicate và object bị bỏ sót                       |
| Instance segmentation | Một class và một polygon/mask cho mỗi instance      | Biên người, tóc, quần áo/dây tạp dề và các vùng tiếp xúc hoặc che khuất khó xác định chính xác                      | Tạo polygon riêng cho từng instance và bám phần biên quan sát được theo guideline             | Kiểm tra polygon có lẫn nền, hở biên, gộp/tách sai instance và xử lý vùng mơ hồ có nhất quán không    |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: chỉ sử dụng các ảnh công khai đã được bài lab phê duyệt và có thông tin nguồn/license; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab/GitHub công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý, không tải lên hoặc chia sẻ tiếp, và báo cho Lab Coach/GV phụ trách để được hướng dẫn.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
