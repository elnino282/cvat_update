# CODEX TASK — Safe Adaptive Vehicle Size Refinement for CVAT

## 1. Vai trò và mục tiêu

Bạn là Senior 3D Computer Vision Engineer và CVAT Maintainer.

Repository: https://github.com/elnino282/cvat_update

Nhánh mục tiêu ban đầu: `feat/lidar-autofit-ux-mock`. Kiểm tra branch, HEAD và working tree thực tế trước khi sửa.

**Nhiệm vụ:** Audit, research, implement, test và nghiệm thu tính năng cải thiện kích thước cuboid Auto Fit cho ô tô LiDAR. Hiện một số box bị ngắn do bề mặt xe chỉ được quan sát một phần.

Giữ nguyên SNAP-Outdoor làm model phân đoạn chính. Cải thiện Geometric Fitting bằng adaptive vehicle size prior và phương pháp mở rộng box theo bằng chứng hình học.

**Mục tiêu quan trọng nhất: Kéo dài box hợp lý mà không fit nhầm xe bên cạnh.**

Không dừng ở bản kế hoạch hoặc pseudocode. Phải sửa mã nguồn thật và chạy các test phù hợp nếu môi trường cho phép.

## 2. Quy tắc bắt buộc — Evidence First

1. Không bịa thuật toán, API, hàm, đường dẫn, checkpoint, dataset, cấu hình, kết quả test hoặc benchmark.
2. Audit source hiện tại trước khi implement; không giả định code giống commit cũ.
3. Mỗi phát hiện quan trọng phải kèm file, dòng code và bằng chứng thực tế.
4. Phân biệt rõ:
   - VERIFIED: Đã kiểm chứng từ source hoặc test.
   - PROPOSED: Giải pháp đề xuất cần kiểm nghiệm.
   - BLOCKED: Không thể xác minh vì thiếu dữ liệu, thiết bị hoặc môi trường.
5. Không tự nhận thuật toán lấy trung bình kích thước là thuật toán gốc của Autoware/OpenBox nếu chỉ lấy cảm hứng từ chúng.
6. Không khẳng định kết quả cải thiện khi chưa có dữ liệu so sánh.
7. Không tự tải model lớn, huấn luyện lại, thay SNAP hoặc thay đổi Docker production không cần thiết.
8. Không ghi đè thay đổi chưa commit của người dùng. Không tự push hoặc merge.
9. Khi mượn code bên ngoài, phải xác minh license và attribution. Ưu tiên tự triển khai bằng NumPy dựa trên nguyên lý đã công bố.
10. Đừng chạy toàn bộ CVAT test suite. Chỉ chạy các test trực tiếp liên quan đến task.

Nếu có giả định chưa được chứng minh, hãy nêu rõ và thiết kế thử nghiệm để xác minh.

## 3. Nghiên cứu các phương pháp có thật

### A. Autoware — Shape Estimation

Nguồn:

https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_shape_estimation/

https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_shape_estimation

Kiểm tra mã nguồn thực tế của:

- L-shape fitting.
- Vehicle size filtering.
- Reference shape size correction.
- `correctWithDefaultValue()`.
- Quy tắc điều chỉnh tâm box khi hiệu chỉnh dimensions.

Phân tích phần nào chuyển được sang Python/NumPy mà không cần ROS.

### B. OpenBox — NeurIPS 2025

Paper và mã nguồn:

https://oliver0922.github.io/OpenBox/

https://github.com/oliver0922/OpenBox

Nghiên cứu:

- Adaptive bounding box generation.
- Visibility-based box extension.
- Statistical object size priors.
- One-sided versus two-sided extension.

Lưu ý: Pipeline OpenBox thực tế còn dùng dữ liệu đa frame, 2D tracking và các thành phần khác. Không được giả định Auto Fit một frame có đầy đủ các tín hiệu đó.

Chỉ áp dụng phần nguyên lý có thể thực hiện và xác minh với dữ liệu hiện có. Nếu cần camera calibration, sensor origin hoặc tracking mà task chưa cung cấp, báo BLOCKED cho phần đó, không tự suy đoán.

## 4. Giai đoạn A — Audit thực tế

Kiểm tra tối thiểu:

- `components/vehicle_fit/geometry.py`
- `components/vehicle_fit/service.py`
- `components/vehicle_fit/snap_backend.py`
- `components/vehicle_fit/tests/`
- `components/vehicle_fit/evaluation/`
- Cấu hình runtime, Compose và biến môi trường liên quan.

Xác định:

1. SNAP tạo mask và chọn instance bằng seed thế nào.
2. `_fit_vehicle()` tính length, width, height, yaw và center ra sao.
3. Candidate scoring, quantile, surface residual, contamination và prior penalty ảnh hưởng thế nào đến box ngắn.
4. `estimate_vehicle()` xử lý partial mask và geometry-only fallback thế nào.
5. `fit_vehicle()` phân loại confident/estimated thế nào.
6. `allowEstimated` có đang thay đổi nhãn chất lượng hoặc đường inference không.
7. Service hiện dùng giá trị `VehiclePriors` nào trong thực tế.

Đặc biệt kiểm tra sự khác biệt đã quan sát ở phiên bản trước:

- Geometry dataclass: 4.20 / 1.80 / 1.55 m.
- Runtime environment defaults: 3.70 / 1.61 / 1.49 m.

Không mặc định tăng chiều dài prior trước khi biết nguyên nhân.

Tái hiện ít nhất một trường hợp box ngắn và ghi lại length ban đầu, prior thực tế, observed footprint, score, quality và hình học box nếu có dữ liệu.

## 5. Giai đoạn B — Implement Adaptive Size Refinement

### B1. Chỉ kích hoạt khi có bằng chứng thiếu quan sát

Không sử dụng `confidence` thấp làm điều kiện duy nhất.

Xem xét kết hợp:

- Observed longitudinal extent.
- Tỷ lệ quan sát theo chiều dài và chiều rộng.
- Số lượng và phân bố điểm hỗ trợ.
- Bằng chứng về cạnh hoặc mặt xe đã quan sát.
- Độ ổn định của yaw và vị trí tâm.
- Mức độ phụ thuộc vào vehicle prior.
- Khả năng mask bị lẫn xe bên cạnh.

Nếu box được hỗ trợ tốt bởi các mặt xe đã quan sát, ưu tiên giữ nguyên.

### B2. Adaptive Vehicle Size Prior

Triển khai một cơ chế hiệu chỉnh có kiểm soát, có thể bắt đầu từ:

L_new = (1 - alpha) * L_fit + alpha * L_prior

Đây là phương án đề xuất để thử nghiệm, không phải công thức được xác nhận là nguyên bản của Autoware hoặc OpenBox.

Yêu cầu:

- Alpha phải dựa trên bằng chứng quan sát, không chỉ confidence.
- Không tự động kéo dài mọi xe đến cùng một kích thước.
- Không thu nhỏ xe có bằng chứng rõ ràng là dài hơn prior.
- Không hardcode một kích thước trung bình KITTI thành quy tắc toàn cục.
- Ưu tiên sử dụng prior hiện có hoặc prior phân phối được thống kê từ tập tune.
- Không lấy ground-truth của tập eval để fit hoặc chọn tham số.
- Các ngưỡng mới phải có lý do, unit test và bằng chứng đánh giá.

Có thể so sánh cải tiến trực tiếp trong candidate scoring với post-fit refinement. Chọn phương pháp đơn giản hơn nếu hiệu quả và an toàn tương đương.

### B3. Visibility-Aware Extension

Không kéo dài box đối xứng theo mặc định.

Nếu có bằng chứng đáng tin rằng một đầu xe đã quan sát rõ còn đầu kia bị khuất:

- Giữ cạnh hoặc mặt đã quan sát.
- Mở rộng về phía thiếu dữ liệu.
- Tính lại center chính xác theo yaw và delta length.
- Giữ đúng hệ tọa độ LiDAR.
- Kiểm tra lại toàn bộ hỗ trợ hình học.

Nếu không thể xác định hướng bị khuất đủ chắc chắn, không được tự bịa hướng mở rộng.

Không dùng camera hoặc ego pose nếu chúng chưa được xác minh là có sẵn, đúng frame và đúng hệ tọa độ.

### B4. Chống fit nhầm xe — P0 Safety Gate

Đã có một lỗi thực tế: mask chứa chủ yếu điểm xe bên cạnh và một seed của xe được chọn; fitter vẫn có thể trả box confident của xe bên cạnh.

Phải kiểm tra:

- Seed nằm trong final oriented cuboid với tolerance nhỏ, có cơ sở kỹ thuật.
- Có cụm điểm thật đủ hỗ trợ quanh seed.
- Final box gắn với đúng component/instance.
- Không chấp nhận box chỉ vì một seed đơn lẻ nằm trong mask.
- Không cho phép prior kéo box lấn sang instance lân cận không được hỗ trợ.
- Box mở rộng vẫn phải qua mọi safety gate.

Tất cả đường trả kết quả, gồm confident, estimated, proposed cuboid và geometry-only fallback, đều phải tuân thủ các bất biến an toàn chung.

Nếu nghi ngờ sai instance, trả `no-fit` cùng reason rõ ràng và giữ annotation hiện có. Không biến lỗi sai instance thành estimated box.

### B5. Quality Classification

Tách quality assessment khỏi policy.

Thứ tự đúng:

1. Generate candidate.
2. Validate instance.
3. Refine dimensions khi có bằng chứng.
4. Validate final geometry.
5. Assess confident/estimated.
6. Áp dụng `allowEstimated`.
7. Return hoặc reject.

Cùng một geometry và cùng bằng chứng phải có cùng đánh giá chất lượng, bất kể `allowEstimated` là true hay false.

Nếu strict không cho estimated, reject thay vì đổi quality thành confident.

Không sử dụng `surface_score` làm bằng chứng duy nhất cho độ tin cậy của phần thân xe chưa quan sát.

## 6. Giai đoạn C — Tích hợp an toàn

Giữ nguyên UX hiện tại:

- Draw Cuboid 3D.
- Checkbox Auto Fit.
- Double-click tự động fit.
- Không yêu cầu Preview.
- Không sinh box trùng.
- Undo/Redo và Save/Reload không bị ảnh hưởng.
- Manual Draw vẫn hoạt động bình thường.
- Timeout, cancellation, frame change, duplicate request và stale result vẫn được xử lý đúng.

Ưu tiên sửa backend geometry/service. Chỉ sửa frontend nếu audit chứng minh có yêu cầu cần thiết.

Giữ API response tương thích. Diagnostics bổ sung phải hữu ích, không chứa dữ liệu nặng không cần thiết.

Ghi tối thiểu khi khả thi:

- Original dimensions.
- Refined dimensions.
- Applied prior.
- Estimated observed coverage.
- Extension direction và cơ sở chọn hướng.
- Quality, review reason.
- Provider: SNAP hoặc geometry-only.
- Safety gate rejection reason.

Có thể dùng feature flag để so sánh trước/sau và rollback. Không bật mặc định nếu acceptance gates chưa đạt.

## 7. Giai đoạn D — Test có mục tiêu

### D1. Synthetic Regression — bắt buộc

Thêm deterministic tests cho:

1. Xe được quan sát đầy đủ: không tự kéo dài vô lý.
2. Xe thiếu phần đuôi: chỉ mở rộng đúng hướng nếu đủ bằng chứng.
3. Xe thiếu phần đầu: giữ đúng cạnh đã quan sát.
4. Xe có cả hai đầu chưa rõ: không tự khẳng định hướng mở rộng.
5. Hai xe đậu sát nhau: không tạo box của xe bên cạnh.
6. Mask chủ yếu thuộc xe B nhưng seed thuộc xe A: không chấp nhận box sai.
7. Seed ngoài final cuboid: reject.
8. Seed đơn lẻ nhưng thiếu supporting cluster: reject.
9. LiDAR thưa và occlusion mạnh: estimated hoặc no-fit theo bằng chứng.
10. Cùng candidate với `allowEstimated=true/false`: đánh giá quality nhất quán.
11. Geometry-only fallback: không bypass safety.
12. Cancellation/timeout: không publish kết quả muộn.

Test phải có assertion thực sự. Không dùng test chỉ kiểm tra code không crash. Không cho phép test PASS chỉ nhờ tắt thuật toán mới.

### D2. KITTI Evaluation

Trước tiên kiểm tra dữ liệu nào thật sự có trong môi trường.

Ưu tiên:

- KITTI 3D Object có nhãn để đo độ chính xác.
- KITTI RAW 50 frame hiện có để kiểm tra các ca thực tế và hành vi UI.

Sử dụng tune/eval split đã đóng băng nếu hiện có.

Không benchmark trên KITTI RAW như thể nó có cùng hệ nhãn Object Detection nếu chưa xác minh GT tương ứng.

Không thay đổi prior hoặc threshold dựa trên tập eval.

Nếu dùng evaluator có calibration, xác minh hash các tài sản liên quan trước khi tính IoU.

Nếu dữ liệu không có sẵn, hoàn tất synthetic regression và báo KITTI BLOCKED; không tạo số liệu giả.

### D3. Metrics trước/sau

So sánh baseline với phiên bản mới trên cùng dữ liệu, seed và protocol:

- Mean/median absolute length error (m).
- Dimension error.
- Center error (m).
- Yaw error.
- Oriented IoU 3D.
- IoU >= 0.70.
- Wrong-instance accepted count.
- Seed-outside accepted count.
- Correct-instance coverage.
- Estimated/confident/no-fit counts.
- Latency p50/p95 khi có điều kiện đo.

Báo cả các case không trả box, tránh làm đẹp IoU trung bình bằng cách chỉ giữ các ca dễ.

Không gộp HTTP failure thành algorithmic no-fit.

Nếu benchmark không chạy hết, phải lưu được bằng chứng đã hoàn thành và ghi rõ phần chưa chạy.

## 8. Acceptance Gates

**Gate P0 — Safety**

- 0 wrong-instance accepted trong bộ adversarial synthetic regression.
- 0 seed-outside accepted.
- Không bypass safety qua estimated hoặc fallback.
- Không tạo annotation sai hoặc trùng khi request bị lặp.

**Gate P1 — Geometry**

- Cải thiện có thể đo được với các case box bị ngắn.
- Không làm hỏng các case xe đã fit tốt.
- Không kéo dài box chỉ để đạt chiều dài trung bình.
- Không đổi sai yaw, center hoặc frame coordinates.
- Trường hợp thiếu bằng chứng phải estimated hoặc no-fit, không confident giả.

**Gate P2 — Evaluation/Integration**

- Unit và targeted regression tests PASS.
- API behavior tương thích.
- Nếu chạy được KITTI: có before/after report và số liệu thật.
- Nếu môi trường đủ điều kiện: kiểm tra deadline tương tác hiện hành; không khẳng định đạt 2 giây nếu chưa đo thực tế.
- Không có thay đổi không liên quan tới scope.

Nếu P0 không đạt: task FAIL, không nghiệm thu dù chiều dài box đã tốt hơn.

Nếu hình học không cải thiện trên tập eval: không bật refinement mặc định; báo rõ kết quả và nguyên nhân.

## 9. Phạm vi và cách thực hiện

Thực hiện theo thứ tự:

1. Audit repository và ghi baseline SHA.
2. Research Autoware/OpenBox, xác nhận khả năng áp dụng.
3. Reproduce các lỗi và viết failing tests.
4. Implement bản sửa tối thiểu nhưng đầy đủ safety.
5. Chạy targeted tests.
6. So sánh baseline/new trên dữ liệu có sẵn.
7. Sửa các regression phát hiện được.
8. Tổng hợp báo cáo cuối cùng.

Không thay toàn bộ pipeline SNAP.

Không triển khai OpenBox end-to-end, 2D tracking, shape completion model hoặc các dependency nặng chỉ vì chúng có trong paper.

Không thay đổi các unrelated UI/backend modules.

## 10. Deliverables

Khi hoàn thành, cung cấp:

**A. Audit findings**
- Root cause.
- File/line.
- Evidence.
- Các giả định đã loại bỏ.

**B. Implementation**
- Danh sách file sửa.
- Thuật toán thực sự áp dụng.
- Vì sao chọn cách này thay vì phương án khác.
- Config/flag và cách rollback.

**C. Verification**
- Lệnh test đã chạy thật.
- Kết quả PASS/FAIL/BLOCKED.
- Before/after metrics.
- Trường hợp vẫn chưa giải quyết.
- Tác động latency, memory nếu đã đo được.

**D. Documentation**
- Cập nhật README hoặc tài liệu kỹ thuật phù hợp.
- Chú thích nguồn Autoware/OpenBox nếu sử dụng ý tưởng.
- Hướng dẫn kiểm tra trực tiếp trên CVAT `localhost:8080` nếu stack đang sẵn sàng.
- Không tự ý rebuild toàn bộ Docker stack production.

**E. Final Decision**
- PASS: đủ bằng chứng nghiệm thu.
- PARTIAL: một phần cải thiện nhưng chưa đạt các gate.
- FAIL: còn lỗi nghiêm trọng.
- BLOCKED: không đủ môi trường hoặc dữ liệu.

Không tuyên bố hoàn thành toàn bộ nếu mới chỉ compile hoặc chỉ chạy synthetic tests.

## 11. Yêu cầu cuối cùng

Tập trung vào chất lượng thực tế của công cụ annotation.

Không tối ưu để tất cả xe đều có box dài hơn. Tối ưu để box khớp với **chiếc xe cụ thể được annotator chọn**.

Nguyên tắc ưu tiên:

**Correct Instance > Valid Geometry > Accurate Dimensions > Coverage > Speed**

Nếu thiếu bằng chứng, ưu tiên từ chối an toàn thay vì tạo một box trông hợp lý nhưng sai thực tế.

Bắt đầu audit ngay, tự triển khai các bước có thể thực hiện, chạy targeted tests và xuất báo cáo có bằng chứng. Không dừng lại chỉ để đưa ra kế hoạch.
