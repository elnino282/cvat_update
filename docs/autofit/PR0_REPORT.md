# PR0 — Audit, baseline và dataset tooling

Ngày 2026-10-16; PRD nội dung **1.1**, mười quyết định giữ nguyên. Worktree bắt đầu
sạch tại `403631ed1c7b9ec694ce61f8f6f2c61a056be425`, branch `feat/auto-fit-pedestrian`.
Không stash/reset/commit, refactor fitter, đổi taxonomy, migration task, camera
inference hay multi-frame. Raw `~/Datasets/nuScenes` chỉ được đọc. Có **một correction
testability nhỏ** trong SNAP adapter: tắt nested pooling shuffle còn sót, được đo
trước/sau trên container tách biệt; service đang chạy không được redeploy.

**PR0 có baseline tái lập cho service đã sửa nhỏ: 100/100 Car giống nhau qua hai lượt
rebuild encoder cache.** Trước sửa, cùng 100 Car thay đổi 89 mask và hai quyết định
no-fit/estimated. Mọi lượt trước/sau đều được lưu, không chọn output thuận lợi để thay
baseline. Baseline mới vẫn chỉ có 7% coverage; quality/estimated-first và full UI
acceptance còn chưa đạt. PR1–PR5 chưa được triển khai.

## Tài liệu và artifact

| Deliverable | Đường dẫn |
| --- | --- |
| UI → API → SNAP → fitter → apply/history/persist; kế hoạch file-level PR1–PR5 | [pr0-code-path.md](pr0-code-path.md) |
| Inventory, mapping, strata, prior counts và hạn chế | [pr0-dataset.md](pr0-dataset.md) |
| Playwright hai project, fixture API, coverage PW-01–18 và runbook | [pr0-playwright.md](pr0-playwright.md) |
| Initial worktree, source/container hash parity | [worktree-audit.json](../../artifacts/autofit-eval/worktree-audit.json) |
| Full manifest, 404 frame và 14.432 target records | Local generated/ignored; [hướng dẫn sinh lại và kiểm hash](../../artifacts/autofit-eval/README.md) |
| Inventory / dev-only prototypes | [inventory](../../artifacts/autofit-eval/nuscenes_inventory.json), [priors](../../artifacts/autofit-eval/nuscenes_priors.json) |
| Byte reproducibility của dataset artifacts | [nuscenes_reproducibility.json](../../artifacts/autofit-eval/nuscenes_reproducibility.json) |
| Selection khóa trước inference | [held-out-selection.json](../../artifacts/autofit-eval/held-out-selection.json) |
| Raw request/response/seed của bốn lượt GPU và hai lượt replay | [raw-evidence.tar.gz](../../artifacts/autofit-eval/raw-evidence.tar.gz), [index/hash từng ledger](../../artifacts/autofit-eval/evidence-index.json) |
| Accuracy/strata/latency/denominators/config | [summary](../../artifacts/autofit-eval/baseline-snap/overview.json), [identity](../../artifacts/autofit-eval/baseline-snap/identity.json) |
| Failure gallery dạng token list | [failure-tokens.json](../../artifacts/autofit-eval/baseline-snap/failure-tokens.json) |
| Lượt repeat đúng 100 Car | [repeat summary](../../artifacts/autofit-eval/baseline-snap-repeat/overview.json), [comparison](../../artifacts/autofit-eval/original-repeatability-check.json) |
| Baseline sau correction, 454 cases + repeat 100 Car | [fixed summary](../../artifacts/autofit-eval/baseline-fixed-snap/overview.json), [fixed repeat summary](../../artifacts/autofit-eval/baseline-fixed-snap-repeat/overview.json), [exact comparison](../../artifacts/autofit-eval/fixed-repeatability-check.json) |
| Replay observed masks riêng cho CPU fitter | [runbook và evidence](../../artifacts/autofit-eval/frozen-mask-replay/README.md) |
| Test results và môi trường đo | [verification](../../artifacts/autofit-eval/verification.json), [execution-environment](../../artifacts/autofit-eval/execution-environment.json); full test logs local/ignored |
| Route contract probes, gồm Van/other | [route-probes.json](../../artifacts/autofit-eval/baseline-snap/route-probes.json) |

Manifest SHA256:
`f4b2fe3beac2fab88c731d08e57487863a56360a54e8021fb45792c9368f0a1c`.
Generator chạy hai lần độc lập cho manifest, inventory và priors đều giống byte-for-byte.
Script hash, metadata source hashes, LiDAR hashes và ngày tạo cố định nằm trong artifact.

### Chính sách artifact sau cleanup

Git giữ report/overview, inventory, priors, selections, provenance và comparisons;
sáu raw ledgers, full per-stratum summaries và cold diagnostics được nén chung
thành archive có index kiểm SHA256 từng file gốc.
Manifest 8 MB và test logs không đưa vào Git. Hai lượt replay trung gian `run-1`/
`run-2` được dọn; cặp final `run-3`/`run-4` cùng mọi lượt GPU trước/sau vẫn được giữ.
Không sửa số đo hoặc hash lịch sử để khớp một commit mới. Runbook mô tả extract và
regenerate; các lệnh replay/compare bên dưới cần khôi phục raw ledgers trước.

Manifest hiện chứa dataroot tuyệt đối. Hai lần sinh lại chứng minh byte determinism
tại vị trí đã audit; chưa chứng minh cùng hash khi chuyển sang đường dẫn máy khác.
Đổi vị trí cần provenance/manifest version riêng, không tự thay hash acceptance.

## Kết quả dữ liệu

Chỉ có **v1.0-mini**, 10 scene, 404 sample, 18.538 source annotation, 31.206 sensor
files. Có 404 LIDAR_TOP keyframe, 3.531 LiDAR sweeps, 6 camera, calibration và ego poses;
P0 chỉ đưa keyframe LiDAR hiện tại vào fitter. Không thiếu file hoặc essential token
reference. Có **75 map→log references ngoài mini subset**, giữ rõ trong inventory;
không dùng các reference này để suy ra calibration còn thiếu.

Split scene cố định seed `20261016`: 6 dev / 2 validation / 2 held_out, không giao nhau
ở scene hoặc instance. Đây là local mini holdout, không phải official nuScenes test.
GT global được chuyển chính xác qua inverse ego pose rồi inverse sensor calibration;
size W/L/H đổi thành L/W/H; full rotation matrix được giữ để tính oriented 3D IoU.
Seed là measured LiDAR return được GT hỗ trợ chọn, không gửi GT box/mask/size cho service.
Không dùng camera visibility làm LiDAR confidence.

| Route | Held-out annotation | Có observed seed | Đo baseline |
| --- | ---: | ---: | ---: |
| vehicles/car | 1.019 | 610 | 100 |
| vehicles/truck | 76 | 57 | 57 |
| vehicles/bus | 141 | 120 | 100 |
| pedestrian | 866 | 762 | 100 |
| two-wheels | 129 | 97 | 97 |
| vehicles/van | Không có GT độc lập | — | Không công bố accuracy |

454 object được chọn round-robin qua strata populated rồi SHA256 rank, trước inference;
không retry seed theo IoU, không nhân bản mẫu để đủ bin. 100 Car đại diện 28 instance,
chưa phải 100 object độc lập. Held-out Truck chỉ 2 instance, Two-Wheels chỉ 4 instance;
validation/held-out không có Two-Wheels no-rider. Các bin trống được ghi `insufficient-data`.
Wilson intervals trong report là khoảng annotation-level; vì lặp track trong hai scene,
không được diễn giải là độ tin cậy cho dân số object độc lập.

**21 prototype records chỉ từ dev**, median/P10/P90/MAD theo median mỗi instance để
tránh track dài chi phối. Car–Van có ba geometry clusters chung; Truck hai cluster;
Bus giữ rigid/bendy nguồn nội bộ; Pedestrian và Two-Wheels giữ pool nguồn/rider nội bộ.
**10/21 records thiếu independent instances** và được đánh dấu. Không gán tên pickup/
tractor cho cluster từ kích thước, không thêm subtype public cho Two-Wheels. Prior chỉ
là artifact, chưa cắm production. P10/P90 chưa phải safety bounds đã được nghiệm thu.

## Baseline real SNAP/fitter trước correction

Endpoint riêng `http://127.0.0.1:8093`, model/checkpoint/config ghi trong identity.
GPU báo NVIDIA RTX 5060 Laptop, Torch 2.7.1+cu128/CUDA 12.8, flash attention;
checkpoint SHA256 `266b00df105b494e9c86d416f4715bf615baf2ba6c830c298f02e3dbebedd8bb`.
Cả năm Python module fitter trong container khớp checkout SHA256. Engine adapter
`/opt/cvat/cvat/apps/engine/auto_fit.py` cũng khớp checkout; không redeploy stack.

Đây là **private-service baseline**, không phải UI latency hoặc nghiệm thu Playwright.
Warm requests dùng prepare-ready frame, budget 1700 ms, `allowEstimated=true`; mỗi
Car đều có `provider=snap`, `cacheHit=true`. Scope service được sinh riêng trong run,
không đọc/sửa task CVAT nào. Các route chưa hỗ trợ được đo contract đúng như hiện tại.

| Route | Requests | Confident / Estimated / Không trả box | Median 3D IoU accepted | HTTP p95 |
| --- | ---: | --- | ---: | ---: |
| Car | 100 | 0 / 6 / 94 | 0,518875 (n=6) | 0,098978 s |
| Truck | 57 | 0 / 0 / 57 unsupported-type | N/A | 0,0117 s |
| Bus | 100 | 0 / 0 / 100 unsupported-type | N/A | 0,0112 s |
| Pedestrian | 100 | 0 / 0 / 100 HTTP 400 invalid label | N/A | 0,0108 s |
| Two-Wheels | 97 | 0 / 0 / 97 HTTP 400 invalid label | N/A | 0,0109 s |
| Van | 0 GT cases; routing probe HTTP 200 unsupported-type | insufficient-ground-truth | N/A | Không là fitting metric |

Latency non-Car ở bảng là thời gian **từ chối**, không phải latency của một fitter
đã triển khai. Không dùng p95 gộp các route này để tuyên bố đạt interactive gate.

Car ở lượt đầu:

- Coverage 6/100 = **6%**; 94 no-fit gồm 53 insufficient-evidence, 41 ambiguous.
- Containment 6/6 trong tolerance **0,05 m**. Correct-instance proxy 6/6 bằng maximum
  oriented GT IoU trên **mọi category neighbor** và containment; chưa có human-curated
  identity truth. Điều này không chứng minh blocking instance gate cho toàn bộ sản phẩm.
- Median BEV IoU 0,582686; median center error 0,516964 m. Chỉ 3/6 center error ≤0,5 m;
  4/6 BEV IoU ≥0,5; 1/6 3D IoU ≥0,7. Median 3D IoU khi quy no-fit thành 0 là **0**.
- Median relative L/W/H errors 16,08% / 14,27% / 3,28%; median yaw error modulo π
  0,056849 rad. Không có front/back GT assessment hoặc yaw uncertainty metadata mới.
- Median lowest-Z error so GT là 0,114687 m, p95 0,352023 m. Đây là box-ground proxy,
  chưa có local road-plane truth để khóa estimated ground tolerance.
- False-confident rate **N/A (0 confident)**, không viết 0% từ mẫu số bằng không.
- HTTP warm p50/p95/p99: **0,016050 / 0,098978 / 0,201819 s** (100 requests, gồm no-fit).
  SNAP stage p50/p95/p99: 0,003601 / 0,004378 / 0,004666 s; backbone cache hit.
- Một unprepared request: **0,274606 s**, n=1. Không phải cold model startup; model đã
  ở trạng thái ready trước run. Prepare-ready p50/p95 0,315725 / 0,320034 s, 57 frame.
  Full model startup và UI timing **chưa đo**. Service chưa tách riêng fit/serialization/
  network; không suy diễn các stage đó từ tổng latency. Data-load có thống kê riêng.

## Repeatability: lỗi trước sửa và bằng chứng sau correction nhỏ

Repeat đúng 100 Car, cùng primary seed, manifest, source/checkpoint/config:
**8 estimated thay vì 6**, 89 masks khác nhau; hai quyết định no-fit→estimated,
hai accepted geometry khác nhẹ. Warm Car p95 repeat 0,100740 s. Khác biệt đầy đủ lưu
ở `original-repeatability-check.json`; không chọn output thuận lợi để thay lượt đầu.

Đã tìm thấy nguyên nhân cụ thể trong SNAP commit
`59fa79b3948e2a9aec4e69efccb1f0677bdb4295`: top-level serialization tắt shuffle nhưng
`SerializedPooling` vẫn mặc định `shuffle_orders=True`, chạy `torch.randperm` trong
eval; constructor không truyền false vào các pooling modules. Rebuild encoder cache
thay đổi masks. Xem source file/line/hash ở code-path report. Chỉ replay cache-hit
trong cùng frame sẽ bỏ sót vấn đề này.

Correction PR0 thêm 17 dòng trong `snap_backend.py`: tắt `shuffle_orders` trên tất cả
named modules có flag đó sau khi load weights, trước eval/CUDA; provenance ghi root
backbone và bốn pooling modules. Không đổi checkpoint, neighborhood, voxel size,
SemanticKITTI condition, fitter/prior hoặc threshold. Hai tests mới khóa sự khác biệt
giữa top-level flag và nested pooling. Đây là inference-order correction có ảnh hưởng
output, được công bố rõ thay vì coi là refactor không đổi behavior.

Container thử nghiệm dùng **cùng image/checkpoint**, mount đúng một file adapter mới
read-only, expose loopback 8094. Service 8093 hiện tại giữ nguyên. Warm-up dùng một
frame dev riêng, sau đó đo cùng selection 454 object và lặp đúng 100 Car/57 frame.
`nuscenes_compare` kiểm manifest, source/config hashes, seed, từng status/quality/reason,
provider, center/size/rotation và maskIndices: **100/100 exact match**, không khác mask
hoặc outcome. UUID và timing không nằm trong equality. Kết luận giới hạn ở corpus/
hardware/version đã đo, không khẳng định mọi CUDA pipeline đều deterministic.

| Car metric | Trước sửa, lượt đầu | Sau correction | Repeat sau correction |
| --- | ---: | ---: | ---: |
| Confident / estimated / no-fit | 0 / 6 / 94 | 0 / 7 / 93 | 0 / 7 / 93 |
| Median 3D IoU accepted | 0,518875 | 0,535569 | 0,535569 |
| Median BEV IoU accepted | 0,582686 | 0,656500 | 0,656500 |
| Median center error m | 0,516964 | 0,233585 | 0,233585 |
| Seed containment / instance proxy | 6/6 / 6/6 | 7/7 / 7/7 | 7/7 / 7/7 |
| HTTP warm p50 / p95 / p99 s | 0,016050 / 0,098978 / 0,201819 | 0,016448 / 0,100687 / 0,264404 | 0,016390 / 0,098570 / 0,256395 |

Sau correction: 52 insufficient-evidence + 41 ambiguous; median relative L/W/H
errors 11,50% / 13,67% / 5,04%; median yaw modulo π 0,090725 rad; lowest-Z error
p50/p95 0,143953/0,335492 m. False-confident vẫn N/A (không confident); all-case median
IoU zero-for-no-fit vẫn 0. Unprepared request 0,257248 s (n=1, không cold model load);
prepare-ready p95 0,318474 s (57 frame); data load p95 0,000937 s. Full per-stratum
metrics, raw responses và failure tokens nằm trong thư mục `baseline-fixed-snap`.
Non-Car vẫn giữ nguyên kết quả unsupported/invalid label, Van chưa có GT riêng.

Median Car IoU tăng 0,016695 so lượt đầu, containment/instance proxy không giảm,
p95 tăng khoảng 1,73% và dưới 2 s. Đây là so sánh mô tả trên corpus đã khóa; baseline
trước sửa stochastic và không có confident output nên không suy ra full quality
acceptance hoặc calibrated false-confident rate. Dùng **bản fixed** làm regression
reference tiếp theo; giữ bản original để review correction.

Frozen observed-mask CPU replay cũng exact-match 100/100 qua hai lượt, giữ outcome
6 estimated/94 rejected từ lượt original; p95 142,979/140,478 ms, không có SNAP/network.
`maskIndices` trên response thành công có thể đã qua fitter filtering; đây không phải
raw SNAP input mask trong mọi record. Replay này là evidence geometry độc lập, không
phải lý do kết luận GPU fixed repeat pass.

## Tests và trạng thái chạy

- Fitter/cloud/service CPU + mocked SNAP: **75 tests, 70 pass + 5 expected failures**.
  17 tests mới pin Car, seed quantile, competing neighbor, coordinate/calibration gaps.
  Expected failures là yêu cầu chưa đạt; unexpected success khiến suite fail.
- Evaluation/tooling: **45 tests pass**, gồm 9 dataset, 6 baseline-tooling,
  7 frozen-replay và 1 repeat-comparison test. Dataset generator chạy hai lần giống byte-for-byte.
- Canvas3d direct Node suite: **17/17 pass**.
- Engine adapter SimpleTestCase trong container: **17/17 pass**, không DB/annotation writes.
- Playwright fixture contracts: **5/5 pass**; discovery **10 tests** (9 mocked + 1 real).
  **Chưa chạy browser, chưa tạo task qua API**, vì chưa có CVAT_USERNAME/PASSWORD.
- Cypress canvas3d **chưa chạy**. PR0 không thay shared production behavior; suite
  được giữ nguyên, không disable/thay bằng Playwright. Chạy ở PR1 khi shared code thay đổi.

Commands chính, từ root repository:

```bash
python3 -m components.vehicle_fit.evaluation.nuscenes_dataset \
  --dataroot "$HOME/Datasets/nuScenes" --version v1.0-mini \
  --seed 20261016 --created-at 2026-10-16 --output /tmp/autofit-pr0-rebuild

python3 -m components.vehicle_fit.evaluation.nuscenes_baseline freeze \
  --manifest artifacts/autofit-eval/nuscenes_manifest.json.gz \
  --split held_out --per-class 100 --output /tmp/autofit-pr0-selection.json

python3 -m components.vehicle_fit.evaluation.nuscenes_baseline run \
  --manifest artifacts/autofit-eval/nuscenes_manifest.json.gz \
  --selection artifacts/autofit-eval/held-out-selection.json \
  --endpoint http://127.0.0.1:8093 --output /tmp/autofit-pr0-new-run

# Sau khi chạy hai lượt cùng source/config/Car seeds, exit 1 nếu có khác biệt:
python3 -m components.vehicle_fit.evaluation.nuscenes_compare \
  --first artifacts/autofit-eval/baseline-fixed-snap \
  --repeat artifacts/autofit-eval/baseline-fixed-snap-repeat \
  --output /tmp/autofit-pr0-fixed-comparison.json

PYTHONPATH=components python3 -m unittest discover -s components/vehicle_fit/tests -v
PYTHONPATH=components python3 -m unittest discover -s components/vehicle_fit/evaluation -t components -v
node cvat-canvas3d/tests/auto-fit.test.cjs

cd tests/playwright-autofit
npm ci --ignore-scripts --workspaces=false
node support/contract.test.mjs
npm test -- --list
```

Output directories phải mới; tooling từ chối overwrite artifact khác. Cần Docker GPU
service đã sẵn sàng để run; commands trong sandbox có thể cần quyền localhost socket.
Service 8093 cũ tái hiện behavior trước correction, không phải source fixed. Để tái
hiện bản fixed mà không redeploy stack, chạy container riêng ở terminal khác:

```bash
docker run --rm --gpus all -p 127.0.0.1:8094:8093 \
  -v "$PWD/components/vehicle_fit/weights/SNAP_Outdoor.pth:/weights/SNAP_Outdoor.pth:ro" \
  -v "$PWD/components/vehicle_fit/snap_backend.py:/app/components/vehicle_fit/snap_backend.py:ro" \
  sha256:515842da1e3bcbe74d5fe0c373f97ecf48dad9c356547d6a720c25d982fe5a46
```

Đợi health có `snapAvailable=true`; prepare một frame dev để warm model trước đo,
dùng endpoint 8094 trong evaluator. Sau lượt 454, chạy lại với
`car-repeat-selection.json` vào output khác; compare như trên. Lệnh tái tạo dựa trên
image đã có ở máy này; máy khác build `Dockerfile.snap` với source/checkpoint đã pin.
Playwright browser prerequisites, exact manifest hash, secrets và lệnh mocked/real nằm
trong runbook riêng. Không lưu password vào artifact.

## Blockers và thứ tự tiếp theo

Đánh giá sau audit: PR0 cung cấp regression reference tái lập và tooling reviewable,
nhưng chưa phải nghiệm thu tính năng multi-class. **7% coverage** chỉ trên 100 Car
seedable được stratify (28 instance), không đại diện toàn bộ 610 Car seedable held-out
hoặc một test set độc lập lớn. Estimated-first là khoảng trống chính cần xử lý ở
shared layer; tốc độ từ chối nhanh không bù được chất lượng/coverage thấp. Không có
confident output nên false-confident vẫn chưa đánh giá được. PR1 cần giữ reference
fixed và xử lý seed support/component/coordinate contracts trước khi mở route mới.

1. **Deployment hiện tại vẫn có pooling shuffle cũ.** Source correction đã được kiểm
   chứng 100/100 exact-repeat trên container riêng, nhưng chưa redeploy/merge. PR1
   phải dùng đúng fixed source/config/manifest reference; bảo toàn original evidence.
2. **Estimated-first chưa đạt**: 93/100 supported Car vẫn bị no-fit sau correction; class routes còn thiếu;
   mixed mask chưa chọn lại seed component. Không hạ safety gate để tăng coverage.
3. **Coordinate contract** thiếu calibration/frame/units, wrong-space seed gate quá muộn;
   finite/Singular/NaN coverage đã ghi thành characterization và expected failures.
4. **Ground/center gates chưa đủ evidence**: baseline original có 3/6 accepted Car vượt center 0,5 m;
   không khóa ground tolerance thuận lợi từ held-out hoặc từ GT lowest-Z proxy.
5. **Van insufficient-ground-truth**: Car–Van pool không tạo được GT Van. Cần metadata/
   fixture curate độc lập mới có Van acceptance; không suy ra subtype từ kích thước.
6. **Mini thiếu diversity**: sparse pools, ít independent Truck/Bus/cycles, thiếu no-rider
   held-out, chưa có trainval/test. Không download thêm hoặc tune trên held-out ở PR0.
7. **Estimated/yaw/prior metadata chưa persist/UI feedback**; UI other còn có thể gọi API;
   Playwright browser/API fixture upload chưa xác minh do thiếu account; Cypress chưa chạy.

Kế hoạch file-level chi tiết nằm ở code-path report, giữ đúng thứ tự:
**PR1 Shared → PR2 Pedestrian → PR3 Vehicles → PR4 Two-Wheels → PR5 P1**.
PR1 phải chốt input/result/prior/metadata contracts và giữ Car regression trước khi
mở route mới. Camera và multi-frame chỉ nằm ở PR5. Mọi class mới giữ experimental
đến khi có real-data gates tương ứng; không báo stable từ mocked tests.
