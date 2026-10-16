# PR0 — Code-path audit và kế hoạch PR1–PR5

Audit ngày 2026-10-16, theo toàn bộ PRD nội dung 1.1, trên checkout bắt đầu tại
`403631ed1`. Tài liệu này mô tả mã trong repository. Runtime Docker phải đối chiếu
hash riêng; không dùng kết quả runtime khác mã để tuyên bố baseline checkout.
PR0 có một correction nhỏ phục vụ tái lập: tắt serialization shuffle còn sót ở
pooling SNAP và ghi module policy vào provenance. Không refactor fitter, đổi
taxonomy/routing hoặc sửa dữ liệu nguồn. Baseline trước sửa được giữ nguyên;
baseline sau sửa chạy trong container riêng, service đang phục vụ ở 8093 không đổi.

## Đường đi thực tế

| Bước | File và vị trí | Hành vi hiện tại |
| --- | --- | --- |
| Checkbox/type | `cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover.tsx:104`; container cùng tên `:186` | Auto Fit chỉ bật cho legacy `Car` hoặc `Vehicles` có immutable select `type` đúng năm giá trị. ID label/attribute lấy từ schema runtime. Type được giữ trong preference theo canvas/job/label. |
| Draw contract | `cvat-ui/src/utils/auto-fit.ts:102` | Tạo identity task/job/frame/label/objectType/vehicleType/typeAttributeId; bật `allowEstimated: true`; chỉ Shape/Track/Cuboid. `isAutoFitIdentityCurrent` ở `:84` kiểm tra frame/tool/label/type hiện hành. Không có pedestrian hoặc two-wheels route. |
| Chuẩn bị frame | `cvat-ui/src/components/annotation-page/canvas/views/canvas3d/canvas-wrapper3D.tsx:644`; `cvat-ui/src/utils/auto-fit-provider.ts:18` | POST/GET `/api/jobs/{runtimeJobId}/auto-fit/prepare`, giữ token trong WeakMap theo canvas, abort khi frame thay đổi; polling chờ encoder. |
| Load/pick | `cvat-canvas3d/src/typescript/canvas3dView.ts:1469`; `autoFitSeed.ts:10` | PCDLoader đọc frame; lọc XYZ không hữu hạn. Picker chỉ raycast LiDAR Points, lấy vertex thật, áp `matrixWorld`, lọc bán kính pixel rồi chọn điểm gần camera nhất. Không dùng giao điểm ray giả làm seed. |
| Double click | `cvat-canvas3d/src/typescript/canvas3dView.ts:1384` | Chặn in-flight đồng bộ; kiểm tra frame/image/deleted/current; hiển thị seed preview; gọi operation đúng một lần. |
| Deadline/dedup | `cvat-canvas3d/src/typescript/autoFit.ts:105` | UUID riêng, snapshot seed/identity, AbortController, deadline 2000 ms, Promise.race chống provider bỏ qua abort; bỏ stale result. `fitResultToPoints` ở `:71` kiểm tra identity/provider/finite/size và trả cuboid 16 số. |
| HTTP provider | `cvat-ui/src/utils/auto-fit-provider.ts:61` | Axios POST `/api/jobs/{id}/auto-fit`, không đi qua cvat-core server proxy cho inference. Deadline provider 1850 ms, không retry tự động; một recovery frame-expired trong deadline; DELETE request khi abort/timeout. |
| Auth/API | `cvat/apps/engine/views.py:2255`; `permissions.py:1004` | JobViewSet dispatch đến adapter; yêu cầu quyền VIEW_DATA và UPDATE_ANNOTATIONS. |
| Validate/load | `cvat/apps/engine/auto_fit.py:43`, `:141`, `:160` | DRF finite seed, UUID, runtime task/job/label/type, 3D/deleted/frame range; đọc compressed PCD giống browser; trả unsupported-type cho non-Car vehicle trước đọc cloud/gọi service. Timeout mặc định 1700 ms (cho phép 1–16000). |
| Private service | `cvat/apps/engine/auto_fit.py:210`; `components/vehicle_fit/service.py:330` | Gửi multipart PCD+metadata hoặc JSON với frameToken đến `/v1/fit`. `FitRuntime` single-flight theo user/task/job/frame/requestId, reject payload mismatch, cancel/deadline/cache giới hạn. Service chỉ trả geometry, không ghi annotation. |
| Cache/frame identity | `components/vehicle_fit/service.py:142`, `:159`, `:178` | Token scope gồm user/task/job/frame/dataVersion; TTL 120 s; tối đa 3 frame/128 MiB; cloud SHA256; encoder cache một frame. Chuẩn bị dùng cùng frame hiện tại, không multi-sweep. |
| Decode/coordinate | `components/vehicle_fit/cloud.py:35`; `service.py:228` | `.bin` hiện là XYZI bốn float32; PCD ascii/binary/LZF. Lọc finite XYZ và giữ inverse original indices. nuScenes raw năm float32 cần conversion rõ ràng trong tooling. Service kiểm tra seed finite; chưa kiểm tra calibration hoặc coordinate-frame metadata. |
| SNAP | `components/vehicle_fit/snap_backend.py:22`, `:66`, `:202` | Direct Python SNAPBackend, không có hop Nuclio trong đường Auto Fit này. Source/checkpoint pinned; CUDA bắt buộc; voxel 0.05 m deterministic với inverse raw mapping; một prompt/mask; sigmoid > 0.5. Correction PR0 tắt nested pooling shuffle và ghi module names; backbone/decoder timing và cacheHit được ghi. |
| Interactive fallback | `components/vehicle_fit/service.py:248`, `:290` | Interactive ≤2000 ms chỉ gọi SNAP khi frame ready; GPU lock tối đa 200 ms. Model/frame/GPU chưa sẵn sàng dùng `estimate_vehicle`, provider geometry-only, estimated policy explicit. Cold preparation không nằm trong interactive fit latency. |
| Ground/instance | `components/vehicle_fit/geometry.py:27`, `:59`, `:112` | Ground RANSAC seed RNG=17; component radius phụ thuộc range. Raw seed phải cách measured return ≤0.02 m. SNAP mask phải chứa seed, ≥55% unique local voxel support, có ≥8 distinct voxels. Kiểm tra extent/connected support nhưng vẫn giữ toàn bộ neural mask nếu nhiều component. |
| Car candidate | `components/vehicle_fit/geometry.py:477`, `:574` | Yaw grid theo residual L-shape + PCA, nhiều length/width/placement hypothesis; chưa có registry hoặc class-specific fitters. Hardcoded Car prior LWH 3.70/1.61/1.49 m, có env override. Geometry và estimated candidate đều qua validator. |
| Safety/quality | `components/vehicle_fit/geometry.py:288`, `:327`, `:399`, `:681` | Bound Car L=2.6–5.6/W=1.45–2.3/H=1.0–2.4 m; oriented seed containment tolerance 0.05 m; reject contamination, joined caps, detached-neighbor overlap; validate trước/sau refinement. Quality dựa face coverage/yaw alternatives; confidence scalar khác quality. |
| Return adapter | `cvat/apps/engine/auto_fit.py:79` | Giữ public schema `status: ok/no-fit` + `quality: confident/estimated`; validate result identity/provider/finite/dimensions. Geometry fallback chỉ hợp lệ khi estimated đã opt-in. Không đổi schema trong PR0. |
| Apply | `cvat-canvas3d/src/typescript/canvas3dView.ts:1416`; `canvas-wrapper3D.tsx:689` | Disable draw đồng bộ, một `canvas.drawn`; wrapper kiểm tra current identity và committed request set; tạo ObjectState với geometry/label/type/frame/objectType. No-fit giữ preview để retry/manual, không tạo annotation mới. |
| History/persist | `cvat-ui/src/actions/annotation-actions.ts:1775`, `:1500`; `cvat-core/src/session-implementation.ts:556`; `annotations-collection.ts:1304`, `:1489`; `annotations-saver.ts:253`, `:385`; `annotations.ts:83` | Đi qua annotation put/create transaction và history CREATED_OBJECTS; Save dùng serverProxy annotations; Reload dùng normal annotation GET. Shape/Track/Undo/Redo không có riêng một đường lưu Auto Fit. |

## Contract và khoảng trống cần giữ rõ

| Route | Checkout PR0 |
| --- | --- |
| `Car`, `vehicles/car` | Cùng một Car fitter; có geometry, SNAP, estimated fallback tùy evidence/readiness. |
| `vehicles/truck`, `vehicles/bus`, `vehicles/van` | API/service trả `unsupported-type` trước decode/SNAP. Chưa có fitter. |
| `vehicles/other` | API/service đã chặn; UI vẫn có thể tạo Auto Fit request rồi nhận no-fit, chưa manual-only ở draw routing. |
| `pedestrian`, `two-wheels` | UI không bật; adapter reject label; private runtime ValueError. Không được báo no-fit coverage thành chất lượng fitter đã triển khai. |
| `Animal`, `Obstacle` | Không có Auto Fit route; manual. |

1. **Coordinate contract còn thiếu:** request chỉ có XYZ/pointIndex; không có coordinateFrame,
   units, transform/calibration provenance hoặc validation finite/invertible của matrix.
   Point index không được service đối chiếu với seed. NaN seed bị chặn trước SNAP;
   finite seed sai không gian chỉ bị fitter từ chối sau khi service đã gọi segment.
   SNAPBackend tự kiểm tra nearest return trước CUDA, nhưng đó chưa phải shared input gate.
   Prepared token sai frame bị chặn trước segment; reason hiện là `invalid-seed`, không phải
   `invalid-coordinates`. Legacy KITTI `evaluation/metrics.py:57` trả cả NaN/singular matrix
   mà chưa validate; PR0 pin lỗi này, không khẳng định production nhận calibration đó.
2. **Estimated-first chưa đạt:** ≥8 unique voxels, compact-patch/yaw/ground gates và
   nhiều `insufficient-evidence` vẫn là terminal. Mask có target cộng neighbor lớn hơn
   bị từ chối ngay cả khi component chứa seed độc lập có thể fit. Không được tăng coverage
   bằng cách bỏ safety gate; PR1 phải chọn lại seed component rồi thử fallback an toàn.
3. **Metadata chưa persist:** `autoFitQuality` chỉ kích hoạt selection sau create.
   `FitOutcome`/ObjectState không giữ diagnostics, yaw_uncertain, prior_version hoặc fit_mode;
   event done có hint rỗng. Chưa thể nghiệm thu Estimated UI/Save/Reload metadata (PW-05/11).
4. **Prior và model condition:** Car vẫn dùng một prior và fixed bounds; chưa dùng dữ liệu
   nuScenes. `snap_backend.py:233` luôn gửi condition `SemanticKITTI`, domain `Outdoor`,
   kể cả nuScenes. Giữ nguyên cho baseline PR0, ghi config; thay condition cần experiment
   riêng và before/after, không âm thầm làm baseline tốt hơn.
5. **Timing/observability:** có overall service, API elapsed, SNAP backbone/decoder;
   chưa tách đầy đủ decode/fitting/serialization/network. Cache warm-up nằm trước click;
   cần ghi cold preparation riêng. Failures chưa luôn có latency field. Confidence không
   đồng nghĩa calibrated confidence (synthetic full Car trả quality confident với scalar
   khoảng 0.082); không dùng scalar thay metric false-confident dựa GT.
6. **GT Van:** nuScenes `vehicle.car` là pool Car–Van, không phải GT Van độc lập.
   Van acceptance phải `insufficient-ground-truth` đến khi có curate độc lập; không đổi
   taxonomy hoặc gán giả subtype để đủ sample count.

### Blocker tái lập real SNAP: pooling vẫn shuffle trong eval

Hai lượt baseline Car với cùng frozen manifest cho output khác nhau (xem
`artifacts/autofit-eval/original-repeatability-check.json`). Kiểm tra **read-only** source
trong container `cvat-vehicle_fit-1` xác định một nguồn nondeterminism cụ thể:

- Adapter trước sửa `components/vehicle_fit/snap_backend.py:132` truyền
  `shuffle_orders=False` vào constructor PointTransformerV3, và provenance
  `serializationShuffle: false` tại `:177` chỉ phản ánh cấu hình cấp này.
- Source upstream **ngoài repository**, `/opt/SNAP/src/ptv3_backbone_domain.py:615`,
  đặt mặc định `SerializedPooling.shuffle_orders=True`; lưu flag ở `:627`.
- PointTransformerV3 tạo từng SerializedPooling ở `:886` mà không truyền flag
  `shuffle_orders` (các arguments ở `:887`–`:891`). Vì vậy top-level false không
  lan xuống pooling.
- Pooling forward `:673`–`:677` gọi `torch.randperm` khi flag true; nhánh này không
  kiểm tra training/eval. `model.eval()` không tắt shuffle này. Top-level forward
  ở `:1156` chỉ dùng flag cho initial serialization.

Provenance của source đã đọc: commit upstream
`59fa79b3948e2a9aec4e69efccb1f0677bdb4295`; SHA256 file
`e11082af0a6d7ef55a9592ac8c7c3a86dd308df722d0290b32f61fde61cd7221`.
Đây là bằng chứng source cho random pooling vẫn hoạt động; chưa khẳng định nó là
nguồn duy nhất của khác biệt CUDA. Muốn chứng minh hết nondeterminism còn phải
kiểm tra nhiều lần rebuild frame, không chỉ decoder trên cache đã warm.

PR0 giữ cả hai raw baseline trước sửa tại `baseline-snap` và `baseline-snap-repeat`.
Frozen observed-mask replay chỉ khóa Car geometry/fitter trên mask đã đo; không
được gọi đó là real-GPU/SNAP E2E pass hoặc dùng latency replay thay interactive latency.

Để giải quyết blocker tái lập ngay trong PR0, correction nhỏ tại
`snap_backend.py:66` duyệt `model.named_modules()` và đặt false cho mọi module có
`shuffle_orders`; gọi sau load checkpoint tại `:180`, trước eval/CUDA. Provenance
`fixedSerializationOrderModules` tại `:194` ghi backbone và bốn pooling modules.
Hai regression tests ở `tests/test_snap_determinism.py` kiểm tra propagation,
idempotence, module list và không sửa weight object. Review không thấy thay đổi
checkpoint parameters, neighborhood size hoặc fitter trong correction này.

Đây là thay đổi inference order nên **baseline sau sửa được đo riêng** trên
container tạm ở 8094, cùng image/weights và một file adapter mount read-only;
service 8093 đang phục vụ không bị đổi. Artifact `baseline-fixed-snap` giữ đo
454 records; lần repeat 100 Car là gate kiểm tra rebuild frame. Kết luận sau
repeat nằm trong `PR0_REPORT.md`; không suy ra E2E pass chỉ từ hai unit tests.
Giữ source/hash trước sửa trong audit và giữ nguyên toàn bộ phép đo cũ. Không chỉ
đặt global random seed rồi coi frame order/process RNG là ổn định.

## Hardcode và dependency audit

- Trong các file production Auto Fit nêu trên, scoped search không thấy allowlist
  task 59/60, job 57 hoặc numeric label 94–98. Router resolve label/type runtime.
  Không có calibration matrix/task-specific calibration trong checkout path này.
  Kết luận không áp dụng tự động cho container đang chạy; xem audit hash runtime riêng.
- `tests/vehicles_qa_router.py:18` tìm latest task tên cố định
  `Vehicles GT regression roundtrip`, user `admin`, frame 109; đây là QA helper phụ thuộc
  lần setup trước. Không dùng làm fixture acceptance nuScenes mới.
- Cypress `auto_fit_box_kitti_raw50_verified.js:19` khóa hai cửa sổ KITTI 104/129,
  nhận archive từ env, tự tạo task/runtime IDs và project seed qua camera. Đây là
  regression KITTI hiện hữu, không phải nuScenes manifest. Giữ nguyên suite.
- `migrate_autofit_vehicles.py:146` nhận task/project scope từ CLI, không gán numeric
  allowlist. Không chạy migration, không đổi taxonomy trong PR0.
- Không dùng giá trị kích thước hoặc threshold hiện có để tạo nhãn GT mới; latent
  size prototypes từ dev chỉ là artifact ở PR0, chưa cắm vào production.

## Characterization/regression đã thêm và chạy

File mới: `components/vehicle_fit/tests/test_pr0_characterization.py` (17 tests).
Geometry dùng code thật; chỉ service ordering tests mock SNAP để deterministic.

- Car golden center/size/yaw, rigid-transform consistency, oriented seed containment.
- Neighbor lớn hơn 20%, gấp đôi returns: seed-connected fitter giữ đúng target;
  mixed SNAP mask hiện từ chối an toàn; một requirement test giữ lỗi thiếu fallback.
- Chín connected bumper points ngoài quantile 99%: validator reject box bị trim,
  toàn pipeline chọn box khác chứa seed thật trong tolerance 0.05 m.
- Non-finite seed/wrong prepared frame không gọi SNAP; finite wrong-space seed hiện
  gọi segment trước rejection; noncar routes không gọi SNAP.
- Pedestrian 5/10/15 returns hiện bị chặn ở routing; NaN/singular legacy calibration
  hiện được parser nhận; valid rigid calibration transform round-trip pass.

Năm `unittest.expectedFailure` là **blocker đã tái hiện**, không phải safety gate pass:
mixed-mask seed fallback, pre-SNAP wrong-space validation, pedestrian attempt, NaN
calibration validation, singular calibration validation. Unexpected success làm suite
fail để người sửa phải bỏ marker và review hành vi. Không disable test cũ.

Lệnh đã chạy:

```bash
PYTHONPATH=components python3 -m unittest vehicle_fit.tests.test_pr0_characterization -v
PYTHONPATH=components python3 -m unittest discover -s components/vehicle_fit/tests -v
```

Kết quả characterization: 17 tests / 12 pass / 5 expected failures. Sau thêm hai
pooling regression tests, full geometry/cloud/service suite có **75 tests / 70 pass /
5 expected failures**; evaluation suite **44 pass**. Lần đầu sandbox chặn socket localhost
của ba HTTP tests; rerun được cấp quyền ngoài sandbox pass. Đây là CPU/mock contract
và synthetic geometry evidence, không thay thế baseline nuScenes/GPU/Playwright.

Các lệnh regression hiện hữu cần giữ:

```bash
node --test cvat-canvas3d/tests/auto-fit.test.cjs
PYTHONPATH=components python3 -m unittest discover -s components/vehicle_fit/evaluation -p 'test_*.py' -v
# Focused engine SimpleTestCases trong container hiện có, không tạo test database:
docker exec cvat_server python -c 'import os,django,unittest; os.environ.setdefault("DJANGO_SETTINGS_MODULE","cvat.settings.production"); django.setup(); suite=unittest.defaultTestLoader.loadTestsFromName("cvat.apps.engine.tests.test_auto_fit"); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
# Trong thư mục tests, với dependencies và CVAT fixture phù hợp:
yarn --immutable
yarn run cypress:run:chrome:canvas3d
```

Những lệnh này được ghi để tái lập; không suy ra đã chạy từ việc có command trong report.
PR0 không sửa shared CVAT/UI behavior; không tuyên bố upstream Cypress pass khi
chưa chạy với đủ browser/server fixture.
Host `python3` có NumPy/SciPy nhưng thiếu Django, Django REST Framework và pytest
(kiểm tra qua `importlib.util.find_spec`). Focused engine tests đã chạy trong
container bằng lệnh trên: **17/17 pass, 0.032 s**. Adapter container
`/opt/cvat/cvat/apps/engine/auto_fit.py` có SHA256
`a4a760de3609b1a6358fc5eaf73876d2ebcf218e1186655615652dc4d9cf6093`,
khớp checkout; thiếu dependency host không còn chặn focused engine tests.
Lệnh `node --test cvat-canvas3d/tests/auto-fit.test.cjs` cũng đã chạy thành công:
TAP báo một file subtest pass, zero failures/skips, 1.601 s. Để hiện từng test,
đã chạy thêm `node cvat-canvas3d/tests/auto-fit.test.cjs`: **17 named tests pass**,
zero failures/skips, TAP duration 445.669 ms. Một intentional-failure probe độc lập
trong `/tmp` xác nhận cả direct mode và `node --test` đều exit 1 khi test fail;
aggregation chỉ giấu chi tiết, không biến failure thành pass. Không suy ra browser
acceptance từ test module Node này.

## Kế hoạch file-level theo thứ tự đã khóa

Đây là đề xuất file mới cho PR sau, không phải file production được tạo ở PR0.
Giữ Car baseline cùng manifest/config hash trước khi di chuyển code; không bắt đầu
refactor lớn nếu baseline còn không tái lập hoặc runtime không khớp checkout.

| PR | File thay đổi/thêm | Phạm vi và gate |
| --- | --- | --- |
| **PR1 Shared** | New `components/vehicle_fit/coordinates.py`, `instance_selection.py`, `fit_result.py`, `class_router.py`, `prior_registry.py`; integrate `service.py`, giữ `geometry.py` Car qua adapter | Coordinate frame/units/calibration runtime, seed support trước SNAP, containment final, chọn component đúng seed, registry đọc versioned dev prototypes, status/reason normalization nội bộ. Car geometry unchanged trừ sửa được regression chứng minh. |
| PR1 API/UI/persist | `cvat/apps/engine/auto_fit.py`, `cvat-canvas3d/src/typescript/autoFit.ts`, `autoFitSeed.ts`, `cvat-ui/src/utils/auto-fit.ts`, `auto-fit-provider.ts`, draw popover và `canvas-wrapper3D.tsx`; metadata xuyên `cvat-core/src/object-state.ts`, `annotations-objects/annotation-common.ts`, `annotations-objects/shape.ts`, `annotations-objects/track.ts`, `annotations-collection.ts`, `cvat/apps/engine/serializers.py` nếu audit schema yêu cầu | Giữ envelope API hiện hữu; runtime schema routing, other manual guard, stable test IDs, Estimated feedback, persist metadata không dùng fake attributes public. Chốt schema metadata trước migration/export; dedup/stale/cancel/save/reload tests. |
| PR1 acceptance | `tests/playwright-autofit/*` scaffold từ PR0; thêm relevant `cvat/apps/engine/tests/test_auto_fit.py`, `cvat-canvas3d/tests/auto-fit.test.cjs` và shared geometry tests | Mocked lifecycle PW-01…14/18; API fixtures riêng resolve task/job/label/attribute IDs; Car before/after cùng held-out, Cypress liên quan. Năm expected failures chuyển dần thành tests pass, không xóa. |
| **PR2 Pedestrian** | New `components/vehicle_fit/fitters/pedestrian.py`, `tests/test_pedestrian.py`; `class_router.py`, `prior_registry.py`, route flags và evaluator/Playwright real cases | Ground-to-top/full-body, robust XY, 1–15-point estimated attempt, seed/neutral yaw_uncertain. Không tự điền age/pose. Real nuScenes và ground/center gates trước stable. |
| **PR3 Vehicles** | New `fitters/car_van.py`, `fitters/large_vehicle.py`, `fitters/vehicle_candidates.py`; `tests/test_large_vehicle.py`, `test_car_van.py`; registry prototype pools, route flags, real fixtures | Car–Van mixture, truck/bus tách pool, min-area/PCA/RANSAC L-shape, observed-side anchoring. Trailer không gộp truck; articulated bus theo source convention/evidence. Van metrics riêng vẫn blocked nếu thiếu curated GT. other manual-only cả UI/API. |
| **PR4 Two-Wheels** | New `fitters/two_wheels.py`, `tests/test_two_wheels.py`; router/runtime ride_state plumbing, registry latent prototype selection, real adversarial fixtures | Một public label, chỉ ride_state. Rider/vehicle spatial merge trong frame, loại pedestrian đứng cạnh. Không thêm bicycle/motorcycle subtype vào UI/export. |
| **PR5 P1** | New `temporal.py`, `object_alignment.py`, `camera_confirmation.py`; bounded cache/lifecycle integration trong `service.py`, stale guard trong provider/operation | Chỉ sau P0 gates; multi-sweep static/dynamic alignment, camera confirmation/yaw và latency/version guards. Không đưa sweeps/camera vào số baseline P0. |

Mỗi PR cần raw JSON + summary/token failures cùng git/config/manifest hashes;
Car median IoU không giảm >0.01, correct-instance/false-confident không xấu hơn,
p95 theo gate PRD. Khóa ground tolerance bằng evidence PR0 và ghi cấu hình trước
acceptance, không chọn ngưỡng từ held-out để làm report pass.
