// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

const canvasSelector = '.cvat-canvas3d-perspective canvas';
const objectSelector = '.cvat-objects-sidebar-state-item';
let localTelemetry502Count = 0;

// The API Event telemetry sink in this local stack returns 502. Record and
// suppress only that sink rejection so annotation save/reload remains tested.
Cypress.on('uncaught:exception', (error) => {
    if (error.message === 'Bad Gateway' && error.stack?.includes('saveEvents')) {
        localTelemetry502Count++;
        return false;
    }
    return undefined;
});
const windows = [
    {
        start: 104, count: 25, fixtureEnv: 'AUTOFIT_RAW50_FIXTURE_A', reportPath: '../components/vehicle_fit/evidence/raw50-task-a-input.json',
    },
    {
        start: 129, count: 25, fixtureEnv: 'AUTOFIT_RAW50_FIXTURE_B', reportPath: '../components/vehicle_fit/evidence/raw50-task-b-input.json',
    },
];

function transformPoint(elements, point) {
    const [x, y, z, w] = point;
    return [
        elements[0] * x + elements[4] * y + elements[8] * z + elements[12] * w,
        elements[1] * x + elements[5] * y + elements[9] * z + elements[13] * w,
        elements[2] * x + elements[6] * y + elements[10] * z + elements[14] * w,
        elements[3] * x + elements[7] * y + elements[11] * z + elements[15] * w,
    ];
}

context('Auto Fit Box real service verified: KITTI RAW consecutive 50 frames', () => {
    const tasks = [];
    const allRuns = [];
    let manifest;
    let checkedNativeHistory = false;

    function createTask(window) {
        const fixturePath = Cypress.env(window.fixtureEnv);
        if (!fixturePath) throw new Error(`Set ${window.fixtureEnv} to its frozen KITTI RAW zip`);
        return cy.readFile(fixturePath, null).then((buffer) => cy.window().then(async (win) => {
            const file = new win.File([buffer], `kitti-drive-0005-${window.start}.zip`, { type: 'application/zip' });
            const name = `Auto Fit KITTI RAW ${window.start}-${window.start + window.count - 1} ${Date.now()}`;
            const task = new win.cvat.classes.Task({
                name,
                labels: [{
                    name: 'Vehicles',
                    type: 'cuboid',
                    attributes: [{
                        name: 'type',
                        mutable: false,
                        input_type: 'select',
                        default_value: 'other',
                        values: ['car', 'truck', 'bus', 'van', 'other'],
                    }],
                }],
                image_quality: 100,
                segment_size: window.count,
                use_zip_chunks: true,
                sorting_method: 'lexicographical',
            });
            const created = await task.save({ clientFiles: [file] });
            return { name, taskId: created.id, jobId: created.jobs[0].id };
        }));
    }

    function visibleSeed(seed) {
        return cy.get(canvasSelector).then(([canvas]) => {
            const pointCloud = canvas.scene.children.find((child) => child.isPoints);
            expect(pointCloud, 'loaded LiDAR frame').not.to.eq(undefined);
            canvas.camera.updateMatrixWorld(true);
            pointCloud.updateMatrixWorld(true);
            const worldTarget = transformPoint(pointCloud.matrixWorld.elements, [...seed.point, 1]);
            const cameraTarget = transformPoint(canvas.camera.matrixWorldInverse.elements, worldTarget);
            const clipTarget = transformPoint(canvas.camera.projectionMatrix.elements, cameraTarget);
            const ndc = clipTarget.slice(0, 3).map((coordinate) => coordinate / clipTarget[3]);
            expect(Math.abs(ndc[2]), 'seed is inside the camera clipping range').to.be.lessThan(1);
            expect(Math.abs(ndc[0]), 'seed is visible horizontally').to.be.lessThan(0.98);
            expect(Math.abs(ndc[1]), 'seed is visible vertically').to.be.lessThan(0.98);
            const rect = canvas.getBoundingClientRect();
            return {
                canvas,
                clientX: rect.left + ((ndc[0] + 1) * rect.width) / 2,
                clientY: rect.top + ((1 - ndc[1]) * rect.height) / 2,
            };
        });
    }

    function startAutoFit(jobId, seed, requestAlias) {
        let requestStartedAt = 0;
        cy.intercept('POST', '**/api/jobs/**/auto-fit*').as(requestAlias);
        cy.interactControlButton('draw-cuboid');
        cy.get('.cvat-auto-fit-box-checkbox input').should('be.enabled').check();
        cy.get('.cvat-auto-fit-mock-indicator').should('not.exist');
        cy.get('.cvat-auto-fit-vehicle-type').click();
        cy.get('.ant-select-item-option').contains(/^car$/).click();
        cy.get('.cvat-draw-cuboid-popover').contains('button', 'Shape').click();
        return visibleSeed(seed).then(({ canvas, clientX, clientY }) => {
            requestStartedAt = Date.now();
            expect(canvas.scene.getObjectByName('drawTemplate'), 'no rough template cuboid').to.eq(undefined);
            let autoFitEvent;
            const captureStatus = (event) => { autoFitEvent = event.detail; };
            canvas.addEventListener('canvas.autofit', captureStatus);
            canvas.dispatchEvent(new canvas.ownerDocument.defaultView.MouseEvent('dblclick', {
                clientX, clientY, bubbles: true,
            }));
            canvas.removeEventListener('canvas.autofit', captureStatus);
            if (autoFitEvent?.status === 'retry' || autoFitEvent?.status === 'cancelled' || !autoFitEvent) {
                return {
                    status: 'no-fit',
                    reason: autoFitEvent?.status === 'cancelled' ? 'cancelled' : 'no-points',
                    noRequest: true,
                    diagnostics: { detail: autoFitEvent?.hint || 'The visible seed did not start an Auto Fit request.' },
                    uiRequestStartedAt: requestStartedAt,
                    uiServiceResponseMs: Date.now() - requestStartedAt,
                };
            }
            return cy.wait(`@${requestAlias}`, { timeout: 22000 }).then(({ request, response }) => {
                cy.get(`@${requestAlias}.all`).should('have.length', 1);
                expect(response, 'authorized CVAT service response').not.to.eq(undefined);
                expect(request.body.jobId).to.eq(jobId);
                expect(request.body.labelId).to.be.a('number');
                expect(request.body.vehicleType).to.eq('car');
                expect(request.body.typeAttributeId).to.be.a('number');
                expect(request.body.seed.pointIndex).to.be.a('number');
                return cy.get(canvasSelector).then(([currentCanvas]) => {
                    const pointCloud = currentCanvas.scene.children.find((child) => child.isPoints);
                    const positions = pointCloud.geometry.getAttribute('position');
                    const index = request.body.seed.pointIndex;
                    expect(index).to.be.within(0, positions.count - 1);
                    const actualPoint = transformPoint(pointCloud.matrixWorld.elements,
                        [positions.getX(index), positions.getY(index), positions.getZ(index), 1]);
                    const coordinateError = Math.hypot(
                        request.body.seed.x - actualPoint[0],
                        request.body.seed.y - actualPoint[1],
                        request.body.seed.z - actualPoint[2],
                    );
                    expect(coordinateError, 'seed XYZ matches the real raycast cloud vertex').to.be.lessThan(1e-5);
                    return {
                        ...response.body,
                        uiRequestStartedAt: requestStartedAt,
                        uiServiceResponseMs: Date.now() - requestStartedAt,
                    };
                });
            });
        });
    }

    function selectCaseForFrame(allCasesByFrame, rawFrame) {
        const candidates = (allCasesByFrame.get(rawFrame) || []).filter((record) => record.seeds?.length);
        if (!candidates.length) return undefined;
        const mode = Number(rawFrame) % 5;
        const neighborDistance = (candidate) => Math.min(...candidates
            .filter((other) => other !== candidate)
            .map((other) => Math.hypot(candidate.gt_center[0] - other.gt_center[0],
                candidate.gt_center[1] - other.gt_center[1])));
        const selected = [...candidates].sort((left, right) => {
            if (mode === 0) return left.range_m - right.range_m;
            if (mode === 1) return right.range_m - left.range_m;
            if (mode === 2) return left.gt_point_count - right.gt_point_count;
            if (mode === 3) return Number(right.occlusion || -1) - Number(left.occlusion || -1);
            return neighborDistance(left) - neighborDistance(right);
        })[0];
        selected.ui_case = ['near', 'farthest-raw', 'fewest-gt-points', 'most-occluded', 'nearest-neighbor'][mode];
        return selected;
    }

    function verifyNativeHistory(taskFrame) {
        if (checkedNativeHistory) return cy.wrap(null, { log: false });
        checkedNativeHistory = true;
        cy.get('.cvat-annotation-header-undo-button').click({ force: true });
        cy.get(objectSelector).should('not.exist');
        cy.get('.cvat-annotation-header-redo-button').click({ force: true });
        cy.get(objectSelector).should('have.length', 1);
        cy.saveJob('PATCH', 200, 'autofitRealSave');
        cy.reload();
        cy.get('.cvat-spinner').should('not.exist');
        cy.goCheckFrameNumber(taskFrame);
        return cy.get(objectSelector).should('have.length', 1);
    }

    function runWindow(windowIndex) {
        const window = windows[windowIndex];
        const task = tasks[windowIndex];
        const inputReport = window.report;
        const rawFrames = Object.keys(manifest.frames).sort()
            .slice(windowIndex * window.count, (windowIndex + 1) * window.count);
        const windowRuns = [];
        let successfulObjects = 0;
        let serviceRequests = 0;
        cy.intercept({ method: 'GET', pathname: `/api/jobs/${task.jobId}` }).as(`loadJob${windowIndex}`);
        cy.visit(`/tasks/${task.taskId}/jobs/${task.jobId}`);
        cy.wait(`@loadJob${windowIndex}`);
        cy.get('.cvat-spinner').should('not.exist');
        cy.get(canvasSelector).should(([canvas]) => {
            expect(canvas.scene.children.some((child) => child.isPoints)).to.eq(true);
        });

        const allCasesByFrame = new Map();
        manifest.cases.forEach((record) => {
            const candidates = allCasesByFrame.get(record.frame) || [];
            candidates.push(record);
            allCasesByFrame.set(record.frame, candidates);
        });
        const processAttempt = (rawFrame, taskFrame, record, index) => {
            const seed = record.seeds[index];
            if (!seed) {
                windowRuns.push({
                    raw_frame: Number(rawFrame),
                    task_frame: taskFrame,
                    case_id: record.case_id,
                    scene_case: record.ui_case,
                    gt_point_count: record.gt_point_count,
                    attempts: index,
                    outcome: 'NO_FIT',
                    annotation_count: windowRuns.filter((entry) => entry.raw_frame === Number(rawFrame) &&
                        entry.outcome === 'CUBOID_CREATED').length,
                });
                return cy.get('body').type('{esc}');
            }
            return startAutoFit(task.jobId, seed, `autofit_${windowIndex}_${taskFrame}_${index}`).then((result) => {
                const entry = {
                    raw_frame: Number(rawFrame),
                    task_frame: taskFrame,
                    case_id: record.case_id,
                    scene_case: record.ui_case,
                    gt_point_count: record.gt_point_count,
                    range_m: record.range_m,
                    occlusion: record.occlusion,
                    attempt: index + 1,
                    seed: seed.point,
                    provider: result.provider,
                    service_request: !result.noRequest,
                    status: result.status,
                    reason: result.reason,
                    latency_ms: result.diagnostics?.latencyMilliseconds,
                    ui_service_response_ms: result.uiServiceResponseMs,
                };
                if (!result.noRequest) serviceRequests++;
                if (result.status !== 'ok') {
                    entry.outcome = result.status === 'no-fit' ? 'SEED_NO_FIT' : 'PROVIDER_ERROR';
                    windowRuns.push(entry);
                    return index < 2 ? processAttempt(rawFrame, taskFrame, record, index + 1) :
                        cy.get('body').type('{esc}');
                }

                expect(result.center).to.have.length(3);
                expect(result.size).to.have.length(3);
                expect(windowRuns.some((sameFrame) => sameFrame.raw_frame === Number(rawFrame) &&
                    sameFrame.outcome === 'CUBOID_CREATED'), 'one annotation maximum for this gesture').to.eq(false);
                cy.get(objectSelector).should('have.length', 1);
                successfulObjects++;
                entry.outcome = 'CUBOID_CREATED';
                entry.annotation_delta = 1;
                entry.ui_commit_ms = Date.now() - result.uiRequestStartedAt;
                windowRuns.push(entry);
                return cy.get(canvasSelector).then(([canvas]) => {
                    const [drawn] = canvas.getDrawnObjects();
                    expect(drawn.state.label.name).to.eq('Vehicles');
                    expect(Object.values(drawn.state.attributes)).to.include('car');
                }).then(() => verifyNativeHistory(taskFrame));
            });
        };

        const processFrame = (taskFrame) => {
            if (taskFrame >= rawFrames.length) return cy.wrap(null, { log: false });
            const rawFrame = rawFrames[taskFrame];
            return cy.goCheckFrameNumber(taskFrame)
                .then(() => cy.get('.cvat-spinner').should('not.exist'))
                .then(() => cy.get(canvasSelector).should(([canvas]) => {
                    expect(canvas.scene.children.some((child) => child.isPoints)).to.eq(true);
                }))
                .then(() => {
                    const record = selectCaseForFrame(allCasesByFrame, rawFrame);
                    if (!record) {
                        const missedCars = allCasesByFrame.get(rawFrame) || [];
                        windowRuns.push({
                            raw_frame: Number(rawFrame),
                            task_frame: taskFrame,
                            case_count: missedCars.length,
                            case_id: missedCars[0]?.case_id,
                            gt_point_count: missedCars[0]?.gt_point_count,
                            outcome: missedCars.length ? 'UNSEEDABLE' : 'NO_CAR',
                            annotation_count: 0,
                        });
                        return null;
                    }
                    return processAttempt(rawFrame, taskFrame, record, 0);
                })
                .then(() => processFrame(taskFrame + 1));
        };

        return processFrame(0).then(() => {
            expect(new Set(windowRuns.map((entry) => entry.raw_frame)).size).to.eq(window.count);
            expect(serviceRequests, 'real inference service requests on this 25-frame window')
                .to.be.greaterThan(0);
            cy.writeFile(`cypress/results/autofit-kitti-raw50-window-${windowIndex}.json`, {
                raw_frames: [Number(rawFrames[0]), Number(rawFrames.at(-1))],
                manifest_sha256: inputReport.frozen_manifest_sha256,
                task_zip_sha256: inputReport.task_zip_sha256,
                frames: windowRuns,
                cuboids_created: successfulObjects,
            });
            allRuns.push(...windowRuns);
        });
    }

    before(() => {
        cy.readFile('../components/vehicle_fit/evidence/raw50-manifest.json').then((frozen) => { manifest = frozen; });
        // The local stack's event-ingest route can return 502; do not let its
        // telemetry failure interrupt annotation-history runtime validation.
        cy.intercept({ method: 'POST', pathname: '/api/events' }, { statusCode: 202, body: {} }).as('cvatTelemetry');
        cy.visit('/auth/login');
        cy.headlessLogin({ nextURL: '/tasks' });
        cy.window().then(async (win) => {
            const ownFixtureName = /^Auto Fit KITTI RAW (?:104-128|129-153) \d{13}$/;
            const previousRuns = await win.cvat.tasks.get({ search: 'Auto Fit KITTI RAW' }, true);
            await Promise.all(previousRuns.filter((task) => ownFixtureName.test(task.name))
                .map((task) => task.delete()));
        });
        createTask(windows[0]).then((task) => { tasks[0] = task; });
        createTask(windows[1]).then((task) => { tasks[1] = task; });
        cy.readFile('../components/vehicle_fit/evidence/raw50-task-a-input.json').then((report) => { windows[0].report = report; });
        cy.readFile('../components/vehicle_fit/evidence/raw50-task-b-input.json').then((report) => { windows[1].report = report; });
    });

    it('uses visible LiDAR seed points and creates one real cuboid on each RAW frame 104-128', function firstRaw25() {
        this.timeout(1800000);
        runWindow(0);
    });

    it('uses visible LiDAR seed points and creates one real cuboid on each RAW frame 129-153', function lastRaw25() {
        this.timeout(1800000);
        runWindow(1);
    });

    after(() => {
        cy.writeFile('cypress/results/autofit-kitti-raw50-ui.json', {
            manifest_sha256: windows[0].report?.frozen_manifest_sha256,
            fixture_sha256: windows.map((window) => window.report?.task_zip_sha256),
            frames: allRuns.sort((left, right) => left.raw_frame - right.raw_frame),
            cuboids_created: allRuns.filter((entry) => entry.outcome === 'CUBOID_CREATED').length,
            suppressed_local_telemetry_502: localTelemetry502Count,
        });
        tasks.forEach((task) => {
            if (!task) return;
            cy.window().then(async (win) => {
                try {
                    const [stored] = await win.cvat.tasks.get({ id: task.taskId });
                    if (stored) await stored.delete();
                } catch (error) {
                    if (!String(error?.message).includes('No Task matches the given query')) throw error;
                }
            });
        });
    });
});
