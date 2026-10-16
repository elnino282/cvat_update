# PR0 — nuScenes inventory, manifest và size prototypes

Ngày tạo: **2026-10-16**. Phạm vi: metadata và LiDAR của `~/Datasets/nuScenes`, chỉ đọc.
Không sửa raw data, không tải thêm dataset, không chạy camera hay multi-sweep inference.
Các camera/sweep được kiểm kê bằng metadata và kiểm tra file tồn tại.

## Artifact và khả năng tái lập

| Artifact | Nội dung |
| --- | --- |
| `artifacts/autofit-eval/nuscenes_inventory.json` | Số lượng, hash metadata, file thiếu, token references, phân phối theo split/class/bin |
| `artifacts/autofit-eval/nuscenes_manifest.json.gz` | Local generated/ignored: 14.432 annotation có mapping, 404 frame, GT/seed/calibration/hash và tất cả neighbor GT; xem `artifacts/autofit-eval/README.md` để sinh lại |
| `artifacts/autofit-eval/nuscenes_priors.json` | 21 prototype records từ dev; median/P10/P90/MAD, số annotation/instance và provenance |

Manifest SHA256: `f4b2fe3beac2fab88c731d08e57487863a56360a54e8021fb45792c9368f0a1c`.
Priors SHA256: `91610855d38745f949f157a2d1a743ff0aced46687d202e5f40caec531322f79`.
Manifest nén khoảng 8,03 MB; không chứa binary point cloud hoặc ảnh.
Python **3.10.12**, NumPy **1.26.4**. Script version: `nuscenes-pr0-v1`.
Prior version: `nuscenes-pr0-v1-17df8848bc27`.

```bash
python3 -m components.vehicle_fit.evaluation.nuscenes_dataset \
  --dataroot "${NUSCENES_DATAROOT:-$HOME/Datasets/nuScenes}" \
  --version v1.0-mini \
  --seed 20261016 \
  --created-at 2026-10-16 \
  --output artifacts/autofit-eval

python3 -m unittest components.vehicle_fit.evaluation.test_nuscenes_dataset -v
```

Generator từ chối ghi đè artifact có nội dung khác. Để so sánh một lần chạy mới,
dùng `--output /tmp/cvat-pr0-dataset-repro` rồi so sánh SHA256 cả ba file.
`--created-at` được khóa trong command; gzip dùng `mtime=0`; JSON sắp xếp key.
Đã chạy lại generator đầy đủ vào `/tmp/cvat-pr0-dataset-repro`: **manifest,
priors và inventory giống byte-for-byte** với lần đầu. Bằng chứng/hash lưu tại
`artifacts/autofit-eval/nuscenes_reproducibility.json`.
Không chọn lại split sau khi đã nhìn metric. Artifact lưu hash toàn bộ 13 bảng metadata,
hash script và hash từng LIDAR_TOP keyframe. `load_cloud()` kiểm tra hash trước khi đọc.
Có thể di chuyển raw dataset bằng tham số `dataroot` mà giữ các đường dẫn tương đối trong manifest.

## Inventory thực tế

- Chỉ có `v1.0-mini`; không có thư mục `v1.0-trainval` hoặc `v1.0-test`.
- 10 scene, 404 sample, 911 instance nguồn, 18.538 annotation nguồn.
- 31.206 sample-data record: tất cả file tồn tại; không có missing sensor/map files.
- LIDAR_TOP: **404 keyframe + 3.531 sweep**, tổng 3.935 file.
- Đủ 6 camera channel, mỗi channel 404 keyframe; có các non-keyframe camera files.
- 120 calibrated-sensor record; 31.206 ego-pose record.
- Có **75 reference `map.log_tokens` trỏ đến log không có trong mini**. Danh sách
  đầy đủ được lưu ở `broken_references` và `nonblocking_map_log_references`.
  Các reference này không nằm trên đường sample → LIDAR_TOP → calibration/ego pose
  của evaluator; không được báo inventory là hoàn toàn không có broken reference.
- Không có broken reference thiết yếu ở sample, annotation, instance, calibration,
  sensor hoặc ego pose. Generator dừng nếu một reference thiết yếu bị hỏng.

`num_lidar_pts` trong metadata được giữ riêng với `gt_point_count` đếm lại từ
single-frame cloud đã transform. 14.431/14.432 annotation mapped khớp chính xác.
Annotation `93d92a0d8e2e480f8e1e1b048e64c67e` có source count 669, đếm lại 670;
manifest dùng containment tolerance `1e-6 m` của evaluator hiện hữu. Đây là chênh lệch
được quan sát, chưa xác định nguyên nhân độc lập; không chỉnh GT để ép khớp.

## Mapping đã khóa

| Category nguồn | Public label/type | Số annotation nguồn | Ghi chú |
| --- | --- | ---: | --- |
| `vehicle.car` | `vehicles/car` | 7.619 | Tạo thêm pool prior Car–Van; không tạo GT Van |
| `vehicle.truck` | `vehicles/truck` | 649 | Trailer giữ category riêng, không gộp vào truck |
| `vehicle.bus.rigid` + `vehicle.bus.bendy` | `vehicles/bus` | 353 + 57 | Source subtype chỉ dùng nội bộ |
| `human.pedestrian.*` | `pedestrian` | 5.040 | Không tự điền public `age`/`pose` |
| `vehicle.bicycle` | `two-wheels` | 243 | `ride_state`: 104 rider, 139 no-rider |
| `vehicle.motorcycle` | `two-wheels` | 471 | `ride_state`: 176 rider, 295 no-rider |
| Các category còn lại | Không là target P0 | 4.106 | Giữ GT trong frame để đánh giá neighbor/instance |

`two-wheels` không có public subtype. `source_category` và latent key chỉ là
provenance của tooling. `ride_state` chỉ lấy từ `cycle.with_rider` /
`cycle.without_rider`; thiếu hoặc xung đột attribute thì giữ `null`, không đoán.
Với `human.pedestrian.*`, mapping bao gồm source `personal_mobility` theo PRD;
category này cần xem xét evidence riêng khi nghiệm thu Pedestrian.

nuScenes gộp Car/Van/SUV trong category nguồn. Vì vậy **Van =
`insufficient-ground-truth`** ở PR0; không được công bố acceptance riêng Van từ
`vehicle.car`, kể cả prototype có kích thước giống Van. Cần fixture/metadata curate
độc lập trước khi nghiệm thu riêng Van. Nguồn chính thức:
[nuScenes annotation instructions](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md).

## Coordinate contract và seed

Annotation nguồn có center/rotation trong global coordinates, `size=[width,length,height]`
và quaternion `[w,x,y,z]`. Generator áp dụng:

```text
lidar_to_global = ego_pose @ calibrated_sensor
center_lidar = inverse(lidar_to_global) @ center_global
axes_lidar = inverse(lidar_to_global).rotation @ axes_global
gt_size = [length,width,height]
```

Không bỏ roll/pitch. `gt_rotation_matrix` giữ ma trận đầy đủ; `gt_rotation` tương
thích Three.js Euler XYZ (`Rx @ Ry @ Rz`) đang dùng bởi evaluator CVAT.
Kiểm tra toàn bộ 14.432 box cho sai số reconstruction lớn nhất **9,99e-16**.
Nguồn chính thức: [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md).

Seed chính là measured point gần median của các point trong GT nhất; có tối đa
hai seed phụ theo farthest-point sampling. GT chỉ chọn seed/scoring, không được gửi
box GT vào fitter. Đây là **GT-assisted seed baseline**, không đo chính xác thao tác
click UI. Manifest giữ cả annotation 0-point/không có seed để denominator không bị mất.

`frames[sample_token].annotations` chứa tất cả source-category GT, gồm 4.106 box
không map target. Neighbor bins dùng khoảng cách biên OBB trong global XY ≤1 m và
vertical interval giao nhau. Không lấy component lớn nhất hoặc chỉ so center distance.
Bin này chỉ dùng stratification/evaluation, không đưa GT neighbor vào production fit.

## Split, coverage và các bin thiếu

Seed **20261016**; sort scene tokens, shuffle bằng Python `Random(seed)`, chia
6 dev / 2 validation / 2 held_out. Scene và instance không cắt qua các split.
Đây là local holdout của mini; không phải official nuScenes test split.

| Split | Scene |
| --- | --- |
| dev | scene-0061, scene-0103, scene-0655, scene-0757, scene-0916, scene-1094 |
| validation | scene-0796, scene-1077 |
| held_out | scene-0553, scene-1100 |

| Class | Dev annotation / seedable / instance | Validation annotation / seedable / instance | Held-out annotation / seedable / instance |
| --- | --- | --- | --- |
| Car | 5415 / 4145 / 264 | 1185 / 761 / 88 | 1019 / 610 / 38 |
| Truck | 544 / 465 / 22 | 29 / 26 / 4 | 76 / 57 / 2 |
| Bus | 214 / 197 / 8 | 55 / 47 / 3 | 141 / 120 / 4 |
| Pedestrian | 4009 / 3429 / 193 | 165 / 119 / 11 | 866 / 762 / 23 |
| Two-Wheels | 554 / 474 / 29 | 31 / 25 / 2 | 129 / 97 / 4 |

Mỗi distribution record có count, min/P10/median/P90/max/MAD cho distance,
dimensions và point count, cộng count từng distance/point/visibility/motion/adjacency/rider bin.
Các khoảng: distance `≤20`, `>20–40`, `>40`; observed points `0`, `1–4`, `5–15`, `16–50`, `>50`.
Visibility giữ bốn mức nguồn, không gọi camera inference.

Các giới hạn bắt buộc ghi khi nghiệm thu:

- Held-out Truck chỉ 57 seedable annotation của **2 instance**; không đủ mục tiêu 100.
- Held-out Two-Wheels chỉ 97 seedable annotation của **4 instance**; không đủ 100.
- **Không có no-rider Two-Wheels** trong validation hoặc held-out; tất cả no-rider
  nguồn nằm ở dev với split đã khóa. Không nhân bản hoặc chuyển dữ liệu để làm đẹp coverage.
- Two-Wheels không có moving/static attribute; giữ `motion=unknown` trên toàn bộ
  corpus, không suy ra movement từ `ride_state` hay dùng temporal velocity trong P0.
- Held-out Truck thiếu visibility 2/3. Validation Truck thiếu visibility 3;
  validation Two-Wheels thiếu visibility 2/3; validation Bus thiếu static.
- Chỉ 2 scene held-out và các annotation lặp cùng instance; số annotation không
  tương đương số object độc lập. Confidence interval cần thể hiện sự phụ thuộc này.
- Các class/split có đủ distance, point và adjacent/isolated bins theo dữ liệu hiện có.

`select_cases()` chọn composite-bin round robin, thứ tự trong bin theo hash seed/token,
không nhìn output của inference. Baseline runner lưu selection riêng trước khi chạy.
Số case nhỏ hơn corpus phải báo đúng cohort và bin coverage của selection.

## Dev-only prototypes

Priors chỉ đọc **dev**. Mỗi instance được gom median dimensions trước khi clustering
để object xuất hiện nhiều frame không chi phối kích thước. K-means trong log-size,
khởi tạo deterministic, tối đa 3 cluster, `k=min(3,max(1,floor(instances/8)))`, có
giới hạn số kích thước phân biệt. Pool source-category/rider giữ nội bộ.
Không tune bằng validation/held_out. Source instance tokens, dev-input hash, version,
date và source metadata hashes có trong artifact.

| Pool nội bộ | Prototype | Independent instance count | Median L/W/H (m) |
| --- | --- | --- | --- |
| Car–Van | geometry-00 | 148 | 4,466 / 1,8315 / 1,562 |
| Car–Van | geometry-01 | 115 | 4,758 / 2,024 / 1,85 |
| Car–Van | geometry-02 | 1 | 6,524 / 2,424 / 3,065 |
| Truck | geometry-00 | 13 | 5,375 / 1,998 / 1,914 |
| Truck | geometry-01 | 9 | 7,127 / 2,592 / 3,024 |
| Rigid Bus | geometry-00 | 6 | 12,116 / 3,0215 / 3,4885 |
| Bendy Bus section | geometry-00 | 2 | 9,13 / 2,8575 / 3,299 |
| Adult pedestrian | 3 clusters | 13 / 81 / 86 | Xem đầy đủ artifact |
| Child / construction worker / personal mobility / police | 4 pools | 4 / 7 / 1 / 1 | Xem đầy đủ artifact |
| Bicycle no-rider / rider | 2 pools | 11 / 3 | 1,791 / 0,55 / 1,137; 1,864 / 0,717 / 1,839 |
| Motorcycle no-rider / rider | 2 pools | 12 / 3 | 2,10 / 0,627 / 1,3375; 2,235 / 0,81 / 1,745 |

Car–Van có 3 geometry prototype được ghi thành 3 record Car + 3 record Van dùng
chung thống kê; tổng cộng **21 record**, không phải 21 subtype. `sample_count` là số
annotation và `instance_count` là số độc lập. P10/P90/MAD dùng instance medians.
Có **10/21 record** dưới 8 instance được đánh dấu `insufficient-data`; đó là dữ liệu mô tả,
chưa phải validated production prior. Geometry cluster không chứng minh truck subtype
pickup/rigid/tractor. P10/P90 chưa được nghiệm thu làm safety range.

## Kiểm thử

Suite dataset: **9 tests pass**. Bao gồm transform với cả ego và sensor rotation,
width/length order, roll/pitch Euler round-trip, quaternion invalid, locked taxonomy,
unknown rider, scene isolation, dev-only instance weighting, frozen/hash-verified
assets, OBB adjacency, full fixture có unseedable target và unmapped neighbor,
read-only source verification, missing file/broken token detection và deterministic gzip.

Không sửa production router/fitter hoặc sử dụng priors này cho production ở PR0.
