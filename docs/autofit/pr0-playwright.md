# PR0 — Playwright design và scaffold

Ngày kiểm tra: 2026-10-16. Production baseline: `403631ed1`.
Phạm vi PR0 là fixture/config và scaffold theo API hiện tại; **chưa nghiệm thu browser workflow**.
Phần Playwright không sửa production UI, SNAP, fitter hoặc Cypress; correction
pooling SNAP riêng của PR0 và evidence trước/sau nằm trong `PR0_REPORT.md`. Package độc lập nằm ở
`tests/playwright-autofit`; lockfile riêng không đổi Yarn lock của repository.

## Hai project và đường API thật

| Project | Dữ liệu | Auto Fit | GPU |
| --- | --- | --- | --- |
| `autofit-ui-mocked` | Hai PCD synthetic nhỏ được tạo trong RAM | Intercept `POST /api/jobs/:id/auto-fit`, `POST/GET .../prepare`, `DELETE .../:requestId` | Không cần |
| `autofit-real-nuscenes` | Một LIDAR_TOP từ manifest nuScenes đã khóa hash | API thật, bắt buộc response provider `snap` | Health phải `ready`, backend `snap`, `snapAvailable=true`, có device/CUDA provenance |

Cả hai project dùng Chromium, một worker, không retry; screenshot chỉ khi fail,
trace/video giữ khi fail. Không upload qua UI. CVAT server/database và một UI build có
`CVAT_AUTOFIT_ENABLED=true`, `CVAT_AUTOFIT_BACKEND=snap`, `CVAT_AUTOFIT_MOCK=false`
là prerequisite cho cả hai. Build mock cũ dùng provider in-process, bypass API và
không phù hợp để kiểm tra request payload ở suite mới này.

Source được kiểm tra trực tiếp:

- `cvat-ui/src/utils/auto-fit-provider.ts`: prepare, deadline 1850 ms, cancel, API POST.
- `cvat-ui/src/utils/auto-fit.ts`: runtime label/attribute mapping; chỉ Car/Vehicles được bật trong UI hiện tại.
- `cvat-canvas3d/src/typescript/autoFit.ts`: API trả `status=ok`, `quality=confident|estimated`.
- `cvat-canvas3d/src/typescript/canvas3dView.ts`: picker và `canvas.drawn`.
- `cvat-ui/src/components/annotation-page/canvas/views/canvas3d/canvas-wrapper3D.tsx`: tạo annotation và giữ type runtime.
- `cvat-core/src/server-proxy.ts`: create task, Upload-Start/Multiple/Finish, poll rq_id.

Không giả định API đã chuyển thành `status=confident|estimated`. PR1 phải giữ compatibility
hoặc cập nhật contract cùng test một cách minh bạch.

## Fixture và quyền sở hữu

1. `POST /api/auth/login` bằng `CVAT_USERNAME`/`CVAT_PASSWORD`; cookie session và CSRF
   được truyền sang API context/browser. Không lưu credential trong evidence.
2. Tạo task qua API với taxonomy đã khóa: `vehicles.type`, `pedestrian.age/pose`,
   `two-wheels.ride_state`, `Animal`, `Obstacle`. Không có subtype Two-Wheels.
3. Tên task chứa UUID của run (`autofit-pr0-mocked-*` hoặc `autofit-pr0-nuscenes-*`).
4. Upload PCD, poll request hoàn tất, tìm job bằng `task_id`, lấy label/type IDs từ
   `/api/labels?task_id=...`. Không sử dụng ID task/job/label của người dùng.
5. Fixture thuộc từng test để annotation độc lập. Mocked upload hai cloud synthetic nhỏ;
   real PR0 chỉ có một smoke test/một cloud. Khi thêm nhiều real cases, chuyển caching
   fixture sang worker scope và chia job/frame rõ ràng để không upload lại cloud lớn.
6. Trước teardown phải GET lại task và so **cả ID lẫn tên UUID chính xác**. Không tìm/xóa
   task theo prefix, không dọn run cũ, không sửa task sẵn có. Nếu process bị kill, evidence
   `task-ownership.json` giúp người vận hành dọn đúng task; không có broad cleanup tự động.
7. Evidence ghi task/job/label/type IDs, manifest/cloud hashes, annotation token, health;
   real test lưu request/response/timing. Không ghi raw cloud hay camera image vào log.

`loadRealFixture` đọc gzip/JSON, kiểm SHA256 toàn manifest, chỉ chọn held-out Car có seed,
sort theo annotation token, hoặc nhận `AUTOFIT_CASE_ID` cụ thể trong tập đó. Cloud path
phải nằm trong dataroot kể cả sau resolve symlink; hash file, stride float32 Nx5 và seed
XYZ/index đều được kiểm tra. PCD XYZ giữ thứ tự vertex, không lọc/reindex NaN âm thầm.
Dữ liệu gốc luôn read-only. Camera browser được đặt theo fixture XYZ, seed screen point
được project qua matrixWorld/camera thật; không hardcode pixel.

Mặc định thiếu prerequisite là failure. Local có thể đặt `AUTOFIT_ALLOW_LOCAL_SKIP=1`
để skip rõ lý do khi thiếu dữ liệu/GPU/biến bắt buộc. Flag này **không có tác dụng** nếu
`CI` tồn tại hoặc `AUTOFIT_ACCEPTANCE=1`. Sai assertion không chuyển thành skip.
CVAT auth/network failure sau preflight vẫn fail để không che lỗi fixture.

## Coverage dự kiến và khoảng trống đã biết

“Có scaffold” dưới đây chưa có nghĩa là browser test đã chạy/pass.

| PRD case | PR0 | Hoàn thiện |
| --- | --- | --- |
| PW-01 | Helper bật Auto Fit; chưa có toggle/manual test riêng | PR1 |
| PW-02 | 4 mocked tests kiểm payload car/truck/bus/van và runtime IDs | PR1 thêm other/manual guard |
| PW-03/04 | Shape/Track: một request, một annotation, geometry | PR1 browser acceptance |
| PW-05 | Partial: estimated envelope tạo một cuboid | PR1 feedback Estimated + persisted metadata; UI hiện tại chỉ activate object |
| PW-06 | No-fit route tests không tạo annotation mới | PR1 phải kiểm giữ seed; seed hiện tại là preview, chưa persisted annotation |
| PW-07 | Mock response reason timeout và UI hồi phục | PR1 thêm network deadline/cancel/late-result |
| PW-08 | Giữ response, chuyển frame, release response, không commit | PR1 browser acceptance |
| PW-09 | Repeated double-click trong pending tạo một request | PR1 thêm out-of-order nhiều request và stale identity |
| PW-10/11/12 | Shape và Track Undo/Redo/Save/Reload geometry/type | PR1 thêm metadata persistence |
| PW-13 | Chưa port Copy/Paste; Cypress hiện hữu vẫn giữ nguyên | PR1 |
| PW-14 | Real Car smoke: prepare/warm, projected seed, SNAP response, một annotation | Baseline accuracy/latency dùng evaluator riêng; smoke không chứng minh full gate |
| PW-15 | Chưa có Truck/Bus/Van acceptance | PR3; Van thiếu GT subtype độc lập |
| PW-16 | Chưa có Pedestrian acceptance | PR2 |
| PW-17 | Chưa có Two-Wheels/rider/adjacent pedestrian acceptance | PR4 |
| PW-18 | Chưa có manual-only guard acceptance; code hiện tại vẫn gửi other | PR1 routing + PR3 vehicle acceptance |

PR1 nên thêm `data-testid` cho checkbox/status/notification. PR0 dùng CSS class có thật
và không thay production chỉ để thêm selector. Chờ annotation state, API response và
frame input; không dùng retry/sleep để giấu race. Full P0 acceptance phải hoàn tất tất cả
case và quality gates, không chỉ 10 scaffold tests của PR0.

## Commands

Từ repository root:

```bash
cd tests/playwright-autofit
npm ci --ignore-scripts --workspaces=false
node support/contract.test.mjs
npm test -- --list
# Chỉ cần một lần khi máy chưa có Chromium phù hợp:
npx playwright install chromium
```

Export credentials bằng môi trường shell/CI secret trước khi chạy; không ghi vào repo.

```bash
export CVAT_BASE_URL=http://localhost:8080
export CVAT_USERNAME='<fixture account>'
export CVAT_PASSWORD='<secret from environment>'
npm run test:mocked
```

Real fixture dùng hash của artifact đã review/khóa; không tự thay hash sau khi manifest đổi:

Manifest đầy đủ là local generated/ignored sau cleanup repo; sinh lại và kiểm hash
theo `artifacts/autofit-eval/README.md` trước khi chạy real project.

```bash
export NUSCENES_DATAROOT="$HOME/Datasets/nuScenes"
export AUTOFIT_MANIFEST=../../artifacts/autofit-eval/nuscenes_manifest.json.gz
export AUTOFIT_MANIFEST_SHA256=f4b2fe3beac2fab88c731d08e57487863a56360a54e8021fb45792c9368f0a1c
export AUTOFIT_GPU_HEALTH_URL=http://127.0.0.1:8093/health
AUTOFIT_ACCEPTANCE=1 npm run test:real
# Local opt-in absence skip, never acceptable as CI evidence:
AUTOFIT_ALLOW_LOCAL_SKIP=1 npm run test:real
```

Suite giả định personal workspace, cùng CVAT origin cho API/UI. Organization-specific
fixtures cần thêm org params đồng bộ với UI trước khi sử dụng; không tự chọn organization.
Không cấu hình webServer tự đổi/redeploy stack hiện tại. Dev UI có thể dùng config
`tests/autofit-webpack.config.cjs` đã có; phải build với real provider flags ở trên.

## Kết quả kiểm tra thật trong PR0

- Cài thành công package `playwright@1.64.1` vào package riêng, tạo `package-lock.json`.
  Không cài browser hoặc đổi stack.
- `node support/contract.test.mjs`: **5 tests pass**, 0 skip. Kiểm cleanup ownership,
  taxonomy runtime/no Two-Wheels subtype, CI skip guard, binary PCD, manifest/cloud hash/seed.
- `npm test -- --list`: **10 tests được discover** trong đúng 2 project (9 mocked, 1 real).
  Đây là discovery, không phải browser execution.
- Loader đọc thành công manifest thật hash ở trên và cloud read-only: case
  `005ac6ffe7b14185afc038dae0960137`, held_out, seed index `15797`, PCD `416781` bytes,
  cloud SHA256 `ca225e0f889a48864900299306988f91028a57a763c2008f5c36197819d54861`.
- **Chưa chạy browser/API task creation/GPU browser smoke**: môi trường không có
  `CVAT_USERNAME`/`CVAT_PASSWORD`. Không suy ra pass từ mocked data/fixture checks.
  Selector, API upload, browser WebGL và projected seed vẫn cần kiểm trên stack có account.
- Không có số latency/accuracy từ Playwright trong PR0. JSON request timing tương lai
  chỉ là smoke evidence; baseline p50/p95/p99 phải từ evaluator và ghi rõ cold/warm/sample count.

Hướng thực hiện tiếp: PR1 Shared hoàn thiện mocked acceptance và metadata; PR2 Pedestrian
thêm real fixture/case; PR3 Vehicles bổ sung Truck/Bus và Van khi có curated GT; PR4 Two-Wheels
thêm rider/neighbor safety; PR5 mới bổ sung camera/multi-frame. Không thay thứ tự này.
