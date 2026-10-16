# PRD — CVAT Multi-Class 3D Auto Fit

**Phiên bản:** 1.1  
**Ngày:** 2026-10-16  
**Trạng thái:** Decision-locked; ready for Codex PR0 audit  
**Repository mục tiêu:** `https://github.com/elnino282/cvat_update`  
**Dữ liệu kiểm thử cục bộ:** `~/Datasets/nuScenes`  
**Nền tảng mục tiêu:** CVAT Community, Docker Compose, Nuclio, GPU NVIDIA RTX 5060 Laptop 8 GB  
**Ngân sách tương tác:** p95 không quá 2 giây sau warm-up  

---

## 1. Tóm tắt điều hành

Chức năng 3D Auto Fit hiện chỉ ổn định với `vehicles.type=car`; các loại còn lại chưa có implementation hoàn chỉnh. Sản phẩm phải chuyển từ một fitter dùng chung sang kiến trúc:

> **SNAP-Outdoor segmentation → seed-anchored instance selection → class-specific fitter → uncertainty-aware prior → confident/estimated/no-fit decision.**

Với class được hỗ trợ, nếu seed có bất kỳ LiDAR point support hợp lệ nào thì hệ thống phải ưu tiên tạo **estimated cuboid** tốt nhất có thể từ hình học quan sát được và size prototype tính từ nuScenes cục bộ. Estimated box phải được đánh dấu rõ để annotator kiểm tra/chỉnh tay. Box chỉ được chấp nhận khi đúng instance, tâm và ground anchor; kích thước nằm trong prior range. Yaw được phép chưa chắc chắn và phải mang `yaw_uncertain=true`. `no-fit` chỉ dùng cho input/calibration không hợp lệ, không có point support, loại không hỗ trợ, request bị hủy/timeout hoặc không thể bảo đảm đúng instance/tâm/ground.

PRD yêu cầu Codex trước tiên audit code, dữ liệu và baseline thực tế; không giả định đường dẫn module, schema API hay số liệu hiện tại chỉ từ mô tả. Mọi thay đổi phải giữ Car ổn định và được nghiệm thu bằng unit/integration tests, CVAT regression tests hiện có và Playwright.

### 1.1 Quyết định sản phẩm đã khóa (10/10)

| # | Chủ đề | Quyết định |
|---|---|---|
| 1 | Taxonomy | Dùng `vehicles` + `type={car,truck,bus,van,other}`, `pedestrian`, `two-wheels`; không hardcode numeric ID |
| 2 | Two-Wheels | Một label duy nhất, chỉ giữ `ride_state`; không lộ subtype bicycle/motorcycle |
| 3 | Thứ tự phát triển | Shared foundation → Pedestrian → Vehicles → Two-Wheels; Car là regression baseline |
| 4 | Estimated policy | Có point support hợp lệ thì luôn thử thả estimated box cho class được hỗ trợ |
| 5 | Sensor P0 | LiDAR + SNAP; camera để P1 |
| 6 | Temporal P0 | Chỉ frame hiện tại; multi-frame để P1 |
| 7 | Size prior | Tính read-only từ `~/Datasets/nuScenes`; cho phép nhiều prototype nội bộ |
| 8 | Estimated gate | Đúng instance/tâm/ground; size trong prior range; yaw có thể uncertain |
| 9 | Playwright data | Task fixture nuScenes riêng; mocked UI + real GPU suite; không phụ thuộc task ID có sẵn |
| 10 | Vehicles P0 | `car`, `truck`, `bus`, `van`; Van dùng prototype Car–Van; `other` không Auto Fit |

---

## 2. Bối cảnh và vấn đề

### 2.1 Hiện trạng

- UX hiện có: annotator bật **Auto Fit Box**, đặt cuboid seed và double-click để gọi inference.
- Pipeline hiện tại sử dụng SNAP-Outdoor để tạo mask và geometric fitting để sinh cuboid.
- Car đã có baseline tương đối ổn định.
- Pedestrian từng gặp các lỗi `invalid-coordinates`, `filter-extent`, mask có điểm nhưng fitter từ chối và nguy cơ picker chọn sai seed.
- Fitter đã từng có lỗi chọn đối tượng bên cạnh dù seed thuộc đối tượng mục tiêu.
- Taxonomy đích là label `vehicles` với immutable select `type={car,truck,bus,van,other}`, label `pedestrian`, và label `two-wheels` với `ride_state`. Label/attribute numeric ID là dữ liệu runtime, không phải contract để hardcode.
- Quy ước `two-wheels`: một label duy nhất, không có subtype public; cuboid bao cả phương tiện và rider/passenger, không bao người đứng cạnh.

### 2.2 Taxonomy contract

| Label | Attribute | P0 Auto Fit |
|---|---|---|
| `vehicles` | immutable select `type`: `car`, `truck`, `bus`, `van`, `other` | Có cho bốn type đầu; `other` manual only |
| `pedestrian` | immutable text `age`, `pose` | Có; Auto Fit không tự bịa giá trị attribute |
| `two-wheels` | immutable checkbox `ride_state` | Có; không thêm subtype |
| `Animal` | Không có | Ngoài phạm vi P0 |
| `Obstacle` | Không có | Ngoài phạm vi P0 |

### 2.3 Nguyên nhân sản phẩm

Một fitter hình hộp cho Car không phù hợp cho mọi class:

- Truck/Bus có chiều dài biến thiên lớn, nhiều subtype và thường chỉ thấy một mặt.
- Pedestrian ít điểm, hình gần đối xứng theo trục đứng nên PCA yaw không ổn định.
- Two-Wheels là vật thể mảnh; xe đạp/xe máy và rider có thể tạo các cụm điểm rời nhau nhưng không được biến thành subtype public.
- LiDAR một frame không đủ bằng chứng cho vật thể xa hoặc bị che khuất.
- Một confidence chung không thể phản ánh mặt nào của cuboid được quan sát và mặt nào được suy đoán.

### 2.4 Cơ sở nghiên cứu

- SNAP hỗ trợ interactive point-cloud segmentation và có checkpoint Outdoor huấn luyện trên SemanticKITTI, nuScenes và PandaSet, nhưng không tự giải quyết cuboid fitting hoặc class semantics: <https://github.com/neu-vi/SNAP>
- RANSAC L-shape fitting là hướng phù hợp để ước lượng cạnh và pose của vehicle trong LiDAR có nhiễu: <https://publications.ri.cmu.edu/l-shape-fitting-based-vehicle-pose-estimation-and-tracking-using-3d-lidar>
- CPD cho thấy size prototype có thể cải thiện box của vật thể LiDAR thưa: <https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Commonsense_Prototype_for_Outdoor_Unsupervised_3D_Object_Detection_CVPR_2024_paper.html>
- Side-aware uncertainty cho thấy cần đánh giá chất lượng từng mặt box thay vì một confidence duy nhất: <https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Not_Every_Side_Is_Equal_Localization_Uncertainty_Estimation_for_Semi-Supervised_ICCV_2023_paper.html>
- 3DAL và DetZero cho thấy temporal/object-centric refinement cải thiện auto-labeling khi một frame quá thưa: <https://openaccess.thecvf.com/content/CVPR2021/html/Qi_Offboard_3D_Object_Detection_From_Point_Cloud_Sequences_CVPR_2021_paper.html> và <https://openaccess.thecvf.com/content/ICCV2023/html/Ma_DetZero_Rethinking_Offboard_3D_Object_Detection_with_Long-term_Sequential_Point_ICCV_2023_paper.html>
- nuScenes quy định rider/passenger nằm trong box Bicycle/Motorcycle nhưng người đứng cạnh không nằm trong box: <https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md>

---

## 3. Mục tiêu

### 3.1 Mục tiêu P0

1. Giữ nguyên hoặc cải thiện chất lượng Car.
2. Hỗ trợ Auto Fit cho `pedestrian`, `two-wheels` và `vehicles.type={car,truck,bus,van}`; `vehicles.type=other` chỉ vẽ/chỉnh tay.
3. Với mọi class được hỗ trợ, ưu tiên tạo estimated cuboid khi có point support hợp lệ; box phải đúng instance/tâm/ground và nằm trong size prior range, còn yaw được phép uncertain.
4. Không tạo box nhầm đối tượng lân cận.
5. Hỗ trợ cả Shape và Track theo UX hiện tại.
6. Đảm bảo Undo, Redo, Save, Reload và chuyển frame không làm tạo box trùng hoặc áp kết quả cũ.
7. P0 chỉ dùng LiDAR + SNAP trên frame hiện tại; nghiệm thu bằng dữ liệu thật trong `~/Datasets/nuScenes` và Playwright.

### 3.2 Mục tiêu P1

- Temporal refinement từ nhiều LiDAR sweeps.
- Object-centric alignment cho vật thể chuyển động.
- Camera-assisted confirmation cho vật thể cực thưa hoặc yaw không xác định.
- Calibration confidence dựa trên held-out validation set.

### 3.3 Ngoài phạm vi P0

- Thay SNAP bằng một detector end-to-end mới.
- Tự động đoán class khi annotator đã chọn `vehicles.type`.
- Camera assistance và multi-frame fitting.
- Batch auto-annotation toàn scene.
- Thay đổi CVAT export format ngoài metadata cần thiết.
- Huấn luyện mô hình lớn mới trước khi fitter và prior baseline được nghiệm thu.

---

## 4. Nguyên tắc nghiệp vụ

### BR-01 — Annotator chọn class

Frontend phải gửi label và attributes runtime cho backend. Router dùng tên/semantic key, không dùng numeric ID. Backend không được âm thầm chuyển `vehicles.type` này thành type khác. `two-wheels` luôn là một label; các prototype xe đạp/xe máy nếu có chỉ là chi tiết nội bộ và không được ghi subtype ra annotation.

### BR-02 — Estimated-first nhưng không được sai instance

Với class được hỗ trợ, mọi seed có point support hợp lệ phải kích hoạt một best-effort fit và mặc định trả `estimated` nếu hình học chưa đủ để trả `confident`. Sai instance vẫn là lỗi blocking: nếu không thể giữ đúng đối tượng, tâm và ground anchor thì trả `unsafe-fit`, không giả vờ confident.

### BR-03 — Estimated không đồng nghĩa với tùy ý

Estimated box phải được thử khi seed có point support hợp lệ và class được hỗ trợ. Kết quả chỉ được áp dụng khi đúng instance, tâm và ground anchor theo tolerance cấu hình; mọi dimension phải nằm trong range của prototype được chọn. Yaw có thể chưa xác định và phải được gắn `yaw_uncertain=true` thay vì loại bỏ box.

### BR-04 — Prior phải đúng taxonomy

Không dùng một kích thước trung bình chung cho mọi đối tượng. Dùng nhiều prototype versioned tính từ nuScenes cục bộ. Prototype có thể phân cụm nội bộ theo hình học/category nguồn và `ride_state`, nhưng không được làm thay đổi taxonomy public. Van dùng pool prototype Car–Van theo quyết định P0.

### BR-05 — Không hardcode task

Không cho phép allowlist task 59/60, job 57, một scene hay một calibration cụ thể. Mọi validation phải dựa trên metadata/calibration thực của task.

### BR-06 — Car là regression gate

Không merge PR làm suy giảm baseline Car vượt ngưỡng cho phép tại mục 12.

### BR-07 — Loại không hỗ trợ

`vehicles.type=other`, `Animal` và `Obstacle` không gọi Auto Fit trong P0. UI phải giữ manual workflow và backend trả `unsupported-type` nếu vẫn nhận request.

---

## 5. Người dùng và luồng chính

### 5.1 Luồng thành công

1. Annotator chọn label/class và Shape hoặc Track.
2. Bật `Auto Fit Box`.
3. Đặt seed cuboid và double-click trên đối tượng.
4. UI phát đúng một request có `request_id` duy nhất.
5. Backend lấy mask SNAP, chọn instance chứa seed, chạy fitter theo class.
6. Backend trả `confident` hoặc `estimated` cuboid.
7. Seed được thay thế/cập nhật thành đúng một cuboid; không tạo duplicate.
8. Estimated box hiển thị thông báo ngắn để annotator kiểm tra/chỉnh tay.
9. Undo/Redo/Save/Reload hoạt động như cuboid CVAT bình thường.

### 5.2 Luồng thất bại

- Nếu request timeout, frame thay đổi, task/job thay đổi hoặc người dùng hủy: bỏ kết quả đến muộn.
- Nếu `no-fit`: giữ seed để chỉnh tay; không tạo cuboid thứ hai.
- Hiển thị reason code thân thiện, không lộ stack trace.

---

## 6. Kiến trúc chức năng yêu cầu

```text
UI double-click
  -> request validation and coordinate normalization
  -> class router
  -> SNAP mask candidates
  -> seed-anchored instance selection
  -> ground/object separation
  -> class-specific candidate fitters
  -> candidate scoring and side observability
  -> geometry/prior fusion
  -> safety validation
  -> FitResult(confident | estimated | no-fit)
```

### FR-01 — Request validation

Hệ thống phải kiểm tra:

- Task/job/frame còn hiện hành.
- Coordinate frame và calibration tồn tại, hữu hạn và invertible.
- Seed có ít nhất một LiDAR point support hợp lệ trong tolerance cấu hình.
- Cặp label/type thuộc danh sách P0 được hỗ trợ.
- Request chưa bị hủy hoặc superseded.

Reason codes tối thiểu:

- `invalid-coordinates`
- `unsupported-type`
- `no-seed-support`
- `unsafe-fit`
- `timeout`
- `cancelled`

`empty-mask`, `insufficient-evidence` và `ground-unavailable` là diagnostic flags để kích hoạt fallback hoặc giải thích `unsafe-fit`; chúng không được tự động biến thành `no-fit` nếu fallback estimated vẫn thỏa gate.

### FR-02 — SNAP mask candidates

- Giữ một thao tác double-click từ người dùng.
- Backend có thể thử nhiều điểm LiDAR lân cận seed hoặc nhiều mask hypothesis nội bộ trong frame hiện tại.
- Không được chọn mask chỉ vì mask đó lớn nhất.
- Lưu các metric: mask point count, seed distance, connected components, spatial extent.

### FR-03 — Instance selection

Candidate hợp lệ phải:

- Chứa seed hoặc có support cluster nối với seed theo ngưỡng class-specific.
- Không bị một component lân cận chi phối.
- Có final cuboid chứa seed với tolerance nhỏ.
- Không vượt quá spatial extent hợp lý của class/prototype.

Nếu hai instance cạnh tranh, ưu tiên component chứa/được neo trực tiếp bởi seed và vẫn thử estimated fit. Chỉ trả `unsafe-fit` khi không thể bảo đảm đúng instance/tâm/ground; không được chuyển sang component lớn hơn chỉ để tăng point count.

### FR-04 — Class router

Routing tối thiểu:

```text
vehicles/car   -> CarFitter (giữ baseline)
vehicles/van   -> CarVanFitter (Car–Van prototypes)
vehicles/truck -> LargeVehicleFitter(truck prototypes)
vehicles/bus   -> LargeVehicleFitter(bus prototypes)
vehicles/other -> unsupported-type; manual only
pedestrian     -> PedestrianFitter
two-wheels     -> TwoWheelerFitter(ride_state, latent prototypes)
Animal         -> out of scope; manual only
Obstacle       -> out of scope; manual only
```

Router phải dựa trên semantic name/attribute value nhận từ task schema runtime; không hardcode các ID `94–98` hoặc attribute ID.

### FR-05 — Prior registry

Prior phải được sinh read-only từ `NUSCENES_DATAROOT` (mặc định `~/Datasets/nuScenes`) bằng training/dev split, được version control và không hardcode rải rác trong fitter. Không dùng held-out acceptance để tạo prior. Cho phép nhiều prototype nội bộ cho một label; việc chọn prototype không được tạo thêm subtype trong UI/export.

Mapping nguồn tối thiểu:

- nuScenes `vehicle.car` tạo pool prototype Car–Van; vì category nguồn gộp car/van/SUV, report phải ghi rõ giới hạn khi đánh giá riêng Van.
- `vehicle.truck` và `vehicle.bus.*` tạo prototype tương ứng.
- `human.pedestrian.*` tạo pedestrian prototypes nếu sample count đủ.
- `vehicle.bicycle` và `vehicle.motorcycle` cùng tạo pool latent cho label `two-wheels`, có thể stratify theo rider evidence/`ride_state`.

Mỗi prior record tối thiểu có:

- Public label/type, latent prototype key và `ride_state` khi có.
- Sample count.
- Median length/width/height.
- P10/P90.
- MAD hoặc dispersion tương đương.
- Dataset/split/version.
- Ngày tạo và script version.

Không dùng validation/test ground truth để xây prior cho chính tập nghiệm thu.

### FR-06 — Geometry/prior fusion

Mỗi dimension dùng trọng số quan sát riêng:

```text
d_final[i] = w[i] * d_geometry[i] + (1 - w[i]) * d_prior[i]
```

- `w[i]` P0 phải đến từ evidence trong frame hiện tại có thể giải thích: số điểm biên, side coverage, occlusion và fit residual. Temporal consistency chỉ được thêm ở P1.
- Clamp theo khoảng hợp lý của prototype; không clamp để che một mask sai.
- Khi một mặt nhìn thấy rõ và mặt đối diện bị khuất, giữ mặt nhìn thấy rồi mở box về phía khuất.
- Không recenter đối xứng làm lệch mặt đã quan sát.

Fallback estimated tối thiểu khi point support rất thưa:

1. Chọn latent prototype gần evidence nhất; nếu không phân biệt được, dùng median của prototype pool đúng public class/type.
2. Neo tâm XY bằng seed/component support; không kéo tâm sang neighbor.
3. Neo đáy Z vào ground estimate cục bộ hợp lệ.
4. Dùng median dimensions và clamp vào P10–P90 của prototype.
5. Dùng hướng seed nếu hợp lệ, nếu không dùng neutral yaw và `yaw_uncertain=true`.

Fallback này là hành vi sản phẩm được yêu cầu, không phải `confident` fit. Nếu không có ground hợp lệ hoặc không thể giữ đúng instance/tâm thì trả `unsafe-fit`.

### FR-07 — Fit result và metadata

Giữ API public hiện hữu nếu có; Codex phải audit trước khi thay schema. Nội bộ cần chuẩn hóa `FitResult` tương đương:

```json
{
  "status": "confident | estimated | no-fit",
  "cuboid": "existing CVAT cuboid representation or null",
  "confidence": 0.0,
  "fit_mode": "geometry | hybrid_prior",
  "yaw_uncertain": false,
  "reason": null,
  "metrics": {
    "point_count": 0,
    "seed_support": 0.0,
    "mask_quality": 0.0,
    "visible_sides": [],
    "prior_version": null
  }
}
```

Không bắt UI hiển thị toàn bộ `metrics`; chúng phục vụ log và đánh giá.

---

## 7. Yêu cầu fitter theo class

### 7.1 Car

- Giữ thuật toán và cấu hình hiện tại làm baseline.
- Chỉ refactor khi có characterization tests khóa hành vi.
- Mọi shared-layer change phải chạy Car regression trước khi mở class mới.

### 7.2 Truck

- Ground removal trước khi BEV fitting.
- Sinh candidate từ robust PCA, minimum-area rectangle và RANSAC L-shape.
- Tách ít nhất các prototype có thể suy ra từ dữ liệu: pickup/light truck, rigid/box truck, tractor cab; Codex phải báo số lượng thật trước khi chốt cluster.
- Không gộp trailer vào truck.
- Khi chỉ thấy một mặt dài: giữ mặt nhìn thấy và dùng length prior có uncertainty.

### 7.3 Bus

- Tách bus prior khỏi truck prior.
- Hỗ trợ rigid bus trước.
- Nếu articulated/bendy bus xuất hiện: ưu tiên từng rigid section theo quy ước dataset; nếu không xác định khớp nối, trả estimated hoặc no-fit theo evidence, không tạo một box cực dài tùy ý.

### 7.4 Van

- Dùng `CarVanFitter` và nhiều prototype trong pool Car–Van; không dùng một average box duy nhất.
- Giữ seed-anchored instance selection và shared vehicle geometry từ Car trước khi fusion với prior.
- Do nuScenes gộp car/van/SUV trong `vehicle.car`, Codex phải báo limitation và không bịa ground-truth subtype. Acceptance riêng Van chỉ được công bố khi có fixture/metadata được curate độc lập.

### 7.5 Pedestrian

- Seed-centered component, robust XY footprint và ground-to-top vertical extent.
- Không dùng PCA yaw làm nguồn duy nhất.
- Yaw P0: geometry cue đủ mạnh → hướng seed nếu hợp lệ → configured neutral yaw với `yaw_uncertain=true`.
- Bao toàn cơ thể và extremities; cho phép estimated height/footprint khi bị che khuất nhưng phải neo ground hợp lệ.
- Phân biệt adult/child prior nếu taxonomy/dữ liệu cho phép; nếu không, dùng mixture/prototype và đánh dấu uncertainty.

### 7.6 Two-Wheels

- Đây là một public label duy nhất `two-wheels`; không xuất hoặc yêu cầu subtype bicycle/motorcycle.
- Box bao vehicle + rider/passenger theo `ride_state`.
- Người đứng cạnh không được gộp.
- Fitter có thể thử các latent prototype học từ bicycle/motorcycle và rider/no-rider của nuScenes, nhưng lựa chọn này chỉ là implementation detail.
- Fitter phải cho phép rider và vehicle tạo hai cụm điểm có khoảng hở nhỏ, nhưng P0 chỉ hợp nhất khi thỏa quan hệ không gian trong frame hiện tại.
- Yaw P0: longitudinal geometry cue → hướng seed nếu hợp lệ → neutral yaw với `yaw_uncertain=true`. Track velocity, multi-frame và camera assistance thuộc P1.

---

## 8. Temporal và camera refinement — P1

- P0 bắt buộc chỉ dùng LiDAR + SNAP của frame hiện tại và phải đáp ứng p95 ≤2 giây; không ngầm lấy sweeps/camera để làm đẹp metric.
- Temporal mode thử bounded context, ban đầu đề xuất `[-2, +2]` sweeps; giá trị cuối phải đo trên RTX 5060.
- Static object: ego-pose compensation.
- Moving object: object-centric alignment; không cộng cloud sau ego compensation đơn thuần vì sẽ smear đối tượng.
- Có thể trả estimated box ngay rồi refine khi an toàn, nhưng không được tự thay đổi annotation sau khi người dùng đã chỉnh mà không có version/request guard.
- Camera-assisted confirmation/yaw là một nhánh P1 độc lập, chỉ triển khai sau khi baseline LiDAR-only được khóa.

---

## 9. Dữ liệu và bộ đánh giá nuScenes

### 9.1 Dataroot

Mặc định:

```bash
export NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-$HOME/Datasets/nuScenes}"
```

Không ghi/chỉnh sửa raw dataset. Mọi manifest, cache và report đặt trong thư mục artifact của repository hoặc thư mục tạm có cấu hình.

### 9.2 Audit dữ liệu bắt buộc

Codex phải chạy script read-only để báo cáo:

- Các version có sẵn: `v1.0-mini`, `v1.0-trainval`, `v1.0-test` nếu có.
- Số scene/sample.
- Sự tồn tại của `samples/LIDAR_TOP`, `sweeps/LIDAR_TOP`, calibration, ego poses và cameras.
- Số annotation theo category nguồn/attribute và mapping sang taxonomy sản phẩm.
- Phân phối `num_lidar_pts`, visibility, distance và dimensions.
- Số Bicycle/Motorcycle nguồn có/không rider để xây latent prototype cho `two-wheels`.
- Thiếu file hoặc broken token references.

Schema chính thức có `translation`, `size`, `rotation`, `num_lidar_pts`, `prev`, `next` và visibility, nên dùng trực tiếp để stratify evaluation: <https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md>

### 9.3 Manifest

Sinh manifest tái lập được, ví dụ:

```text
artifacts/autofit-eval/nuscenes_manifest.json
```

Mỗi record gồm:

- scene/sample/annotation token
- source category/attributes
- `target_label`, `vehicles_type` và `ride_state` sau mapping
- LIDAR_TOP path
- GT cuboid
- distance đến ego
- `num_lidar_pts`
- visibility
- moving/static nếu suy ra hợp lệ
- split (`dev`, `validation`, `held_out`)

Random seed phải cố định và được ghi trong report.

### 9.4 Stratification

Mỗi class phải bao phủ tối đa dữ liệu có sẵn theo các bin:

- Distance: `0–20 m`, `20–40 m`, `>40 m`.
- LiDAR points: `1–4`, `5–15`, `16–50`, `>50`.
- Visibility: bốn mức nuScenes.
- Moving/static.
- Isolated/adjacent-instance.
- Rider/no-rider cho Two-Wheels.

Mapping manifest bắt buộc:

| Category nuScenes | Taxonomy sản phẩm | Ghi chú |
|---|---|---|
| `vehicle.car` | `vehicles/car` và pool prior Car–Van | Không suy ra nhãn Van nếu thiếu metadata curate |
| `vehicle.truck` | `vehicles/truck` | Direct mapping |
| `vehicle.bus.*` | `vehicles/bus` | Giữ source subtype nội bộ nếu cần prototype |
| `human.pedestrian.*` | `pedestrian` | Không ghi đè `age`/`pose` nếu không có evidence |
| `vehicle.bicycle`, `vehicle.motorcycle` | `two-wheels` | Category nguồn chỉ phục vụ latent prototype |

`vehicles/van` cần fixture/metadata curate độc lập để có acceptance metric riêng. Nếu chưa có, vẫn triển khai Car–Van prototype nhưng report phải ghi `insufficient-ground-truth`; không coi `vehicle.car` là ground truth Van.

Nếu dataset hiện có không đủ một bin, report phải ghi `insufficient-data`; không tự nhân bản mẫu để đạt số lượng.

### 9.5 Bộ test ba tầng

1. **Smoke:** 2–5 object/class, dùng khi phát triển.
2. **Dev regression:** tối thiểu 20 object/class nếu dữ liệu đủ.
3. **Held-out acceptance:** mục tiêu ít nhất 100 object/class; nếu class hiếm thì dùng toàn bộ held-out hợp lệ và báo confidence interval/sample count.

Không tune threshold trên held-out acceptance.

---

## 10. Chiến lược kiểm thử

### 10.1 Unit tests

Tối thiểu phải có:

- Coordinate validation và transform round-trip.
- Seed support/containment.
- Connected-component cạnh tranh.
- Ground removal.
- L-shape/min-area/PCA candidate generation.
- Prior lookup, median/P10/P90 và versioning.
- Side observability và geometry/prior fusion.
- Car characterization tests.
- Pedestrian yaw uncertainty.
- Rider–vehicle merge và loại người đứng cạnh.
- Cancellation/request deduplication.

Các regression fixture bắt buộc:

- Mask gồm target seed và một neighbor lớn hơn.
- Seed nằm ngoài final box sau quantile filtering.
- Truck chỉ nhìn thấy một mặt.
- Pedestrian 5–15 điểm.
- Two-Wheels có rider và một pedestrian đứng cạnh; cùng public label nhưng thử nhiều latent prototype nội bộ.
- Calibration NaN/singular/wrong frame.

### 10.2 Integration tests

- Service contract với SNAP mocked để deterministic.
- Real SNAP smoke test có GPU tag riêng.
- API request cho từng `vehicles.type`; `other` phải trả `unsupported-type` mà không gọi SNAP.
- Timeout, cancellation và stale-frame result.
- Persist/reload metadata.
- Dataset evaluator chạy được từ manifest mà không cần CVAT UI.

### 10.3 CVAT regression tests hiện hữu

CVAT upstream hiện hướng dẫn chạy Cypress và suite canvas3d bằng:

```bash
cd tests
yarn --immutable
yarn run cypress:run:chrome:canvas3d
```

Nguồn: <https://docs.cvat.ai/docs/contributing/running-tests/>

Không xóa, disable hoặc thay thế suite này bằng Playwright. Nếu thay đổi shared CVAT behavior, phải chạy các Cypress test liên quan.

---

## 11. Playwright acceptance suite

### 11.1 Mục đích

Playwright là acceptance layer riêng cho workflow Auto Fit. Suite phải dùng Chromium và chạy được với CVAT tại `http://localhost:8080` hoặc `CVAT_BASE_URL`.

Playwright config phải giữ bằng chứng lỗi:

- `screenshot: 'only-on-failure'`
- `trace: 'retain-on-failure'` hoặc `on-first-retry`
- `video: 'retain-on-failure'` khi dung lượng cho phép

Tài liệu chính thức: <https://playwright.dev/docs/test-use-options> và <https://playwright.dev/docs/test-webserver>

### 11.2 Hai project kiểm thử

#### A. `autofit-ui-mocked`

- Chạy nhanh, deterministic.
- Intercept Auto Fit API.
- Không yêu cầu GPU/SNAP/dataset lớn.
- Kiểm tra UI lifecycle, request payload và annotation state.

#### B. `autofit-real-nuscenes`

- Tag `@real-nuscenes` và có thể `@gpu`.
- Dùng dữ liệu từ `NUSCENES_DATAROOT`.
- Một Playwright setup project/API fixture tạo task nuScenes nhỏ, riêng biệt từ manifest đã khóa; ghi nhận task/job ID trả về ở runtime và teardown hoặc reuse theo namespace của run.
- Không phụ thuộc task 59/60, job 57 hoặc task do người dùng tạo sẵn; không upload dataset lớn qua UI cho mỗi test.
- Chạy tuần tự nếu dùng chung GPU/service.
- Skip với lý do rõ ràng nếu dataset hoặc service không có; CI acceptance có dataset thì skip bị xem là failure cấu hình.

### 11.3 Playwright test cases bắt buộc

| ID | Kịch bản | Kết quả mong đợi |
|---|---|---|
| PW-01 | Bật/tắt Auto Fit | Trạng thái đúng, Draw thường không bị ảnh hưởng |
| PW-02 | Chọn từng `vehicles.type` | Payload giữ đúng type; ID được resolve từ task schema runtime |
| PW-03 | Double-click một lần | Đúng một request và một cuboid |
| PW-04 | Response `confident` | Seed trở thành fitted box, không duplicate |
| PW-05 | Response `estimated` | Cuboid được tạo, metadata/status báo Estimated |
| PW-06 | `no-fit` | Giữ seed, không tạo box thứ hai |
| PW-07 | Timeout | Thông báo lỗi, UI không treo |
| PW-08 | Chuyển frame khi pending | Kết quả cũ bị bỏ |
| PW-09 | Request duplicate/out-of-order | Chỉ request mới nhất được áp dụng |
| PW-10 | Undo/Redo | Hành vi nhất quán cho fitted box |
| PW-11 | Save/Reload | Geometry, type và metadata được giữ |
| PW-12 | Shape và Track | Cả hai mode hoạt động |
| PW-13 | Copy/Paste Vehicles box | Attribute `type` gốc không bị ghi đè |
| PW-14 | Car regression | Workflow Car không suy giảm |
| PW-15 | Real Truck/Bus/Van | Có point support thì trả confident/estimated đúng gate; Van dùng Car–Van prototype |
| PW-16 | Real Pedestrian | Có point support thì ưu tiên estimated; đúng tâm và ground |
| PW-17 | Real Two-Wheels | Một public label; bao rider, không nuốt người đứng cạnh |
| PW-18 | `vehicles.type=other` | Không gọi SNAP; giữ manual workflow và trả `unsupported-type` nếu gọi API |

### 11.4 Độ ổn định canvas3d

- Thêm `data-testid` cho checkbox, trạng thái và notification nếu thiếu; không phụ thuộc text dịch chuyển khi có selector ổn định.
- Chuẩn hóa camera/view trước thao tác canvas.
- Không chỉ dùng sleep cố định; chờ network response/state cụ thể.
- Seed screen coordinate phải đến từ fixture/projection helper và calibration của fixture, không hardcode pixel tùy máy nếu có thể.
- Mọi flaky retry phải được điều tra; không tăng retry để che race condition.

---

## 12. Metrics và acceptance gates

### 12.1 Metrics bắt buộc

- Correct-instance rate.
- Seed containment rate.
- 3D IoU và BEV IoU.
- Center translation error.
- Relative length/width/height error.
- Yaw error khi yaw có thể đánh giá.
- Coverage: confident, estimated, no-fit.
- False-confident rate.
- Latency p50/p95/p99 và cold/warm tách riêng.
- Kết quả theo class, distance, point count và visibility bin.

Không chỉ dùng một IoU tổng hợp. nuScenes cũng tách translation, scale và orientation error trong evaluation: <https://www.nuscenes.org/object-detection>

### 12.2 Safety gates — blocking

- Wrong-instance trong adversarial seed/neighbor regression set: **0**.
- Final confident/estimated box phải chứa seed theo tolerance cấu hình: **100%**.
- Với class hỗ trợ, request hợp lệ có bất kỳ seed point support nào phải đi qua estimated/confident attempt: **100%**; không được loại chỉ vì point count thấp.
- Mọi estimated box được áp dụng phải đúng instance, có tâm/ground hợp lệ và từng dimension nằm trong range của prototype đã chọn: **100% trên tập output được chấp nhận**.
- `yaw_uncertain=true` được chấp nhận và không làm box fail; không được gắn yaw confidence giả.
- Invalid coordinates/calibration không được gọi SNAP: **100%**.
- Không duplicate annotation trong PW-03/PW-09: **100%**.
- Undo/Redo/Save/Reload/Shape/Track: tất cả test bắt buộc pass.

### 12.3 Car regression gate

PR0 phải đo baseline hiện tại và lưu artifact. Sau thay đổi:

- Không giảm quá `0.01` median 3D IoU trên cùng held-out set.
- Không giảm correct-instance rate.
- Không tăng false-confident rate.
- p95 không vượt 2 giây hoặc tăng quá 15% so với baseline nếu baseline thấp hơn 2 giây.

Nếu baseline hiện tại không đủ tái lập, đó là blocker cần sửa trong PR0; không được tự đặt baseline thuận lợi sau khi code đã thay đổi.

### 12.4 Multi-class quality gates đề xuất

Các ngưỡng dưới đây là mục tiêu ban đầu và chỉ được điều chỉnh sau PR0 bằng report có dữ liệu:

- `confident`: ít nhất 90% kết quả đạt quality threshold tương ứng class.
- `estimated`: correct instance **100% trên accepted output**; center error ≤0.5 m cho Vehicles và ≤0.35 m cho Pedestrian/Two-Wheels; ground-contact error phải nằm trong tolerance được khóa từ PR0; dimensions nằm trong range của prototype đã chọn.
- Vehicle quality threshold: 3D IoU ≥0.70 khi có đủ support; với sparse estimated dùng BEV IoU ≥0.50 cộng center/dimension gates.
- Pedestrian/Two-Wheels quality threshold: 3D IoU ≥0.50 hoặc BEV IoU ≥0.50 cộng center error ≤0.35 m.
- Không được tăng coverage bằng cách hạ confidence khiến false-confident vượt gate.

Coverage và quality phải báo cáo riêng: estimated-first làm tăng số box được trả về nhưng không hạ gate đúng instance/tâm/ground. Các reason `no-fit` hợp lệ chỉ gồm input/calibration không hợp lệ, không có point support, `unsupported-type`, `cancelled`, `timeout` hoặc `unsafe-fit` khi không thể giữ đúng instance/tâm/ground.

### 12.5 Latency gate

- Warm interactive p95 ≤2.0 giây trên RTX 5060 Laptop 8 GB.
- Timeout phải trả control cho UI và không áp late result.
- Report tách: data load, SNAP, fitting, serialization/network.

---

## 13. Observability

Structured log mỗi request tối thiểu gồm:

- request/task/job/frame ID
- class/type
- seed coordinate frame
- point/mask/component counts
- selected candidate score
- geometry/prior weights
- prior version
- status/reason
- latency breakdown

Không log raw camera images hoặc toàn bộ point cloud mặc định.

Evaluator phải xuất:

- JSON machine-readable.
- Markdown summary.
- Failure gallery hoặc token list để tái hiện.
- Git commit, config hash, dataset version và random seed.

---

## 14. Rollout và feature flags

- `vehicles/car`: stable/default theo behavior hiện tại.
- `pedestrian`, `vehicles/truck`, `vehicles/bus`, `vehicles/van` và `two-wheels`: experimental theo từng route cho đến khi pass gates.
- `vehicles/other`, `Animal`, `Obstacle`: manual only trong P0.
- Flag phải cấu hình được, không hardcode UI theo task ID.
- Nếu một route chưa pass gate, giữ dưới feature flag `experimental`; khi flag tắt thì dùng manual cuboid. Khi flag bật vẫn tuân thủ estimated-first và safety gates; không quảng bá là stable.

---

## 15. Kế hoạch PR

### PR0 — Audit, baseline và dataset tooling

**Phạm vi:** Không thay behavior production trừ testability fix nhỏ được chứng minh.

Deliverables:

- Code-path map UI → API → SNAP → fitter → annotation update.
- Git/worktree audit và danh sách thay đổi liên quan hiện có.
- nuScenes inventory report.
- Reproducible manifest generator.
- Baseline accuracy/latency report cho Car và các class hiện có.
- Characterization tests cho Car và các lỗi đã biết.
- Kế hoạch file-level cho PR1–PR5.

**Gate:** Không bắt đầu refactor lớn nếu chưa có baseline tái lập.

### PR1 — Shared safety and uncertainty layer

- Coordinate validation tổng quát.
- Seed support/containment.
- Competing-component rejection.
- Class router.
- Prior registry interface.
- Standard FitResult/reason codes.
- Playwright mocked suite.

### PR2 — Pedestrian

- Ground-anchored fitter.
- Full-body estimated fallback.
- Yaw uncertainty.
- Real nuScenes evaluation.

### PR3 — Vehicles Truck/Bus/Van

- LargeVehicleFitter và CarVanFitter.
- L-shape/min-area candidates.
- Truck, Bus và Car–Van prototype pools.
- `vehicles/other` manual-only guard.
- Real nuScenes evaluation và Playwright smoke.

### PR4 — Two-Wheels

- Composite rider–vehicle fitter.
- `ride_state` và latent prototype priors; không thêm public subtype.
- Adjacent-pedestrian rejection.
- Real nuScenes evaluation.

### PR5 — Temporal và camera refinement (P1)

- Multi-sweep cache.
- Static/dynamic routing.
- Object-centric alignment.
- Camera-assisted confirmation/yaw.
- Latency and stale-result hardening.

Mỗi PR phải độc lập reviewable, có test, report before/after và không gộp nhiều class chưa nghiệm thu vào một commit lớn.

---

## 16. Definition of Done

Tính năng chỉ được coi là hoàn thành khi:

1. Tất cả safety gates pass.
2. Car regression pass.
3. Unit/integration tests pass.
4. CVAT canvas3d Cypress regression pass khi thay shared behavior.
5. Playwright mocked suite pass ổn định.
6. Playwright real nuScenes suite pass cho class được công bố stable.
7. Accuracy/latency report được lưu và tái lập từ commit/config/dataset manifest.
8. Không có task-specific hardcode.
9. Estimated box có metadata và UI feedback rõ ràng.
10. README/runbook ghi cách chạy evaluator, Playwright và GPU smoke tests.
11. Playwright real suite tự tạo task fixture nuScenes riêng qua API và không phụ thuộc ID có sẵn.

---

## 17. Chỉ dẫn bắt buộc cho Codex

1. Đọc toàn bộ PRD trước khi sửa code.
2. Bắt đầu bằng PR0; không triển khai toàn bộ class trong một lượt.
3. Kiểm tra `git status`, branch/HEAD, thay đổi chưa commit và hướng dẫn repository; không ghi đè code người dùng.
4. Dùng code search để xác định đường đi thật của Auto Fit; không đoán filename/API từ PRD.
5. Chỉ dùng nguồn chính thức/paper/repository gốc khi cần research thêm; ghi link trong audit report.
6. Inventory `~/Datasets/nuScenes` ở chế độ read-only; không download lại nếu dữ liệu đã đủ.
7. Chạy baseline trước khi thay algorithm và lưu raw results.
8. Viết test tái hiện bug trước khi fix khi có thể.
9. Không báo pass chỉ dựa trên mock; class stable cần real-data evidence.
10. Với class được hỗ trợ và có point support hợp lệ, ưu tiên estimated box theo gate đã khóa; chỉ dùng `no-fit` cho các reason được liệt kê tại mục 12.4. Không bịa metric hoặc kết quả test.

### Prompt giao Codex đề xuất

> Đọc toàn bộ `PRD_CVAT_MultiClass_AutoFit_v1.0.md` phiên bản nội dung 1.1 và thực hiện **PR0 — Audit, baseline và dataset tooling** trước. Mười quyết định tại mục 1.1 là đã khóa: không đổi taxonomy, không thêm subtype cho Two-Wheels và không mở camera/multi-frame trong P0. Hãy phân tích code thực tế từ UI đến SNAP/fitter, bảo toàn mọi thay đổi chưa commit, kiểm kê read-only dữ liệu tại `~/Datasets/nuScenes`, dựng manifest đánh giá tái lập, đo baseline Car và các route hiện có, tính nhiều size prototype từ training/dev split, và thêm characterization/regression tests cho seed containment, competing instances và coordinate validation. Thiết kế Playwright gồm project UI mocked và real-GPU với task fixture nuScenes riêng được tạo qua API; không hardcode task/job ID. Research chỉ từ nguồn chính thống khi cần. Chưa refactor production lớn ở PR0. Cuối cùng giao: code-path map, inventory/mapping dữ liệu, baseline accuracy/latency, test commands, danh sách blocker (đặc biệt giới hạn ground truth Van) và kế hoạch file-level đúng thứ tự PR1 Shared → PR2 Pedestrian → PR3 Vehicles → PR4 Two-Wheels → PR5 P1. Không bịa kết quả.

---

## 18. Tài liệu tham khảo chính

1. SNAP official repository: <https://github.com/neu-vi/SNAP>
2. nuScenes paper: <https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html>
3. nuScenes annotation instructions: <https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md>
4. nuScenes schema: <https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md>
5. CVAT running tests: <https://docs.cvat.ai/docs/contributing/running-tests/>
6. CVAT 3D annotation: <https://docs.cvat.ai/docs/annotation/manual-annotation/modes/3d-object-annotation/>
7. Playwright test configuration: <https://playwright.dev/docs/test-use-options>
8. Playwright web server: <https://playwright.dev/docs/test-webserver>
9. CPD size prototypes: <https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Commonsense_Prototype_for_Outdoor_Unsupervised_3D_Object_Detection_CVPR_2024_paper.html>
10. Side-aware localization uncertainty: <https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Not_Every_Side_Is_Equal_Localization_Uncertainty_Estimation_for_Semi-Supervised_ICCV_2023_paper.html>
11. Offboard 3D Auto Labeling: <https://openaccess.thecvf.com/content/CVPR2021/html/Qi_Offboard_3D_Object_Detection_From_Point_Cloud_Sequences_CVPR_2021_paper.html>
12. DetZero: <https://openaccess.thecvf.com/content/ICCV2023/html/Ma_DetZero_Rethinking_Offboard_3D_Object_Detection_with_Long-term_Sequential_Point_ICCV_2023_paper.html>
13. Playwright fixtures: <https://playwright.dev/docs/test-fixtures>
14. Playwright API testing: <https://playwright.dev/docs/api-testing>
15. Playwright projects/dependencies: <https://playwright.dev/docs/test-projects>

