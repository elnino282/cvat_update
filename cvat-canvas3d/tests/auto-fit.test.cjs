// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const ts = require('typescript');
const THREE = require('three');
const vm = require('node:vm');
const path = require('node:path');

// Exercise the actual TypeScript modules without introducing another test framework.
require.extensions['.ts'] = (module, filename) => {
    module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText, filename);
};
const { AutoFitOperation, fitResultToPoints } = require('../src/typescript/autoFit.ts');
const { createMockAutoFitProvider } = require('../src/typescript/autoFitMock.ts');
const { pickAutoFitSeed } = require('../src/typescript/autoFitSeed.ts');
const { Canvas3dModelImpl, Mode, UpdateReasons } = require('../src/typescript/canvas3dModel.ts');
const identity = { taskId: 1, jobId: 2, frame: 3, labelId: 4, objectType: 'shape' };
const seed = { x: 1, y: 2, z: 3, pointIndex: 0 };
const request = { ...identity, seed, requestId: 'fixture', requestedAt: 0 };
const result = {
    status: 'ok', requestId: 'fixture', frame: 3, center: [1, 2, 3],
    size: [4, 1.8, 1.6], rotation: [0.1, 0.2, 0.3], provider: 'mock',
};

test('adapter preserves frame XYZ, Euler radians, x/y/z sizes and 16-value format', () => {
    assert.deepEqual(fitResultToPoints(result, request, 'mock'),
        [1, 2, 3, 0.1, 0.2, 0.3, 4, 1.8, 1.6, 0, 0, 0, 0, 0, 0, 0]);
    assert.deepEqual(request.seed, seed);
});

test('adapter rejects mismatches and every invalid numeric field', () => {
    for (const override of [{ requestId: 'wrong' }, { frame: 4 }, { provider: 'snap' },
        { status: 'unknown' }, { confidence: NaN }, { confidence: 1.1 }]) {
        assert.throws(() => fitResultToPoints({ ...result, ...override }, request, 'mock'));
    }
    for (const field of ['center', 'rotation', 'size']) {
        for (const invalid of [NaN, Infinity, -Infinity, '1', undefined]) {
            for (let axis = 0; axis < 3; axis++) {
                const values = [...result[field]];
                values[axis] = invalid;
                assert.throws(() => fitResultToPoints({ ...result, [field]: values }, request, 'mock'));
            }
        }
    }
    for (const size of [[0, 1, 1], [1, -1, 1], [1, 1, 0], [1, 1], [1, 1, 1, 1]]) {
        assert.throws(() => fitResultToPoints({ ...result, size }, request, 'mock'));
    }
});

test('gesture guard admits one operation synchronously and snapshots input', async () => {
    const op = new AutoFitOperation();
    let release;
    let calls = 0;
    let captured;
    const draw = { identity: { ...identity }, isCurrent: () => true, provider: {
        kind: 'mock', fit: (r) => { calls++; captured = r; return new Promise((resolve) => { release = resolve; }); },
    } };
    const first = op.start(draw, { ...seed });
    assert.equal(op.inFlight, true);
    assert.equal((await op.start(draw, seed)).status, 'cancelled');
    assert.equal(calls, 1);
    draw.identity.labelId = 99;
    release({ ...result, requestId: captured.requestId });
    const outcome = await first;
    assert.equal(outcome.status, 'ok');
    assert.equal(outcome.request.labelId, 4);
    assert.match(captured.requestId, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert.equal(op.inFlight, false);
});

test('cancel and timeout release even an abort-ignoring provider; late responses cannot commit', async () => {
    const op = new AutoFitOperation();
    let release;
    const draw = { identity, isCurrent: () => true, provider: {
        kind: 'mock', fit: () => new Promise((resolve) => { release = resolve; }),
    } };
    const first = op.start(draw, seed);
    op.cancel();
    assert.equal((await first).status, 'cancelled');
    release(result);
    assert.equal(op.inFlight, false);
    assert.equal((await op.start(draw, seed, 5)).status, 'retry');
    assert.equal(op.inFlight, false);
});

test('stale lifetime drops matching success and matching error', async () => {
    for (const mode of ['success', 'reject']) {
        const op = new AutoFitOperation();
        let current = true;
        const fit = op.start({ identity, isCurrent: () => current,
            provider: createMockAutoFitProvider(mode, 5) }, seed);
        current = false;
        assert.equal((await fit).status, 'cancelled');
    }
});

test('an old cancelled request cannot release or overwrite a newer operation', async () => {
    const op = new AutoFitOperation();
    let release;
    const first = op.start({ identity, isCurrent: () => true, provider: {
        kind: 'mock', fit: () => new Promise((resolve) => { release = resolve; }),
    } }, seed);
    op.cancel();
    const second = op.start({ identity, isCurrent: () => true, provider: createMockAutoFitProvider('success', 10) }, seed);
    assert.equal((await first).status, 'cancelled');
    assert.equal(op.inFlight, true);
    release(result);
    assert.equal((await second).status, 'ok');
});

test('destroy leaves the canvas idle and invalidates drawing without emitting a UI cancel during a reducer', () => {
    const model = new Canvas3dModelImpl();
    const reasons = [];
    model.subscribe({ notify: (_model, reason) => reasons.push(reason) });
    model.draw({ enabled: true, autoFit: { identity, isCurrent: () => true, provider: createMockAutoFitProvider() } });
    assert.equal(model.mode, Mode.DRAW);
    reasons.length = 0;
    model.destroy();
    assert.equal(model.mode, Mode.IDLE);
    assert.equal(model.data.drawData.enabled, false);
    assert.equal(model.data.drawData.autoFit, undefined);
    assert.deepEqual(reasons, [UpdateReasons.DESTROY]);
});

test('production/default gate is OFF and explicit real flag never reads mock overrides', async () => {
    const source = fs.readFileSync(path.resolve(__dirname, '../../cvat-ui/src/utils/auto-fit.ts'), 'utf8');
    const compiled = ts.transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    for (const [NODE_ENV, CVAT_AUTOFIT_MOCK] of [['production', 'true'], ['development', 'false']]) {
        class Canvas3d {}
        const canvas = new Canvas3d();
        const job = { id: 2, taskId: 1, dimension: '3d', labels: [{ id: 4, name: 'Car' }] };
        const annotation = {
            job: { instance: job }, player: { frame: { number: 3, fetching: false } },
            drawing: { activeLabelID: 4, activeObjectType: 'shape', activeShapeType: 'cuboid' },
            canvas: { instance: canvas, activeControl: 'draw-cuboid' },
        };
        const exports = {};
        const context = {
            exports, process: { env: { NODE_ENV, CVAT_AUTOFIT_MOCK } },
            CVAT_AUTOFIT_MOCK_BUILD_ENABLED: CVAT_AUTOFIT_MOCK === 'true',
            CVAT_AUTOFIT_REAL_BUILD_ENABLED: false,
            CVAT_AUTOFIT_BACKEND_KIND: 'geometry-only',
            window: { localStorage: { getItem: () => assert.fail('Production read test overrides') } },
            require(name) {
                if (name === 'cvat-canvas3d-wrapper') return {
                    Canvas3d, createMockAutoFitProvider: () => assert.fail('Ungated mock construction'),
                };
                if (name === 'cvat-core-wrapper') return {
                    DimensionType: { DIMENSION_3D: '3d' }, ShapeType: { CUBOID: 'cuboid' },
                };
                if (name === 'reducers') return { ActiveControl: { DRAW_CUBOID: 'draw-cuboid' } };
                if (name === 'cvat-store') return { getCVATStore: () => ({ getState: () => ({ annotation }) }) };
                if (name === './auto-fit-provider') return {
                    createRealAutoFitProvider: (kind) => ({
                        kind,
                        async fit(r) {
                            return { status: 'no-fit', requestId: r.requestId, frame: r.frame,
                                reason: 'unavailable', provider: kind };
                        },
                    }),
                };
                assert.fail(`Unexpected import ${name}`);
            },
        };
        vm.runInNewContext(compiled, context);
        exports.setAutoFitPreference(canvas, true);
        assert.equal(exports.autoFitAvailable, false);
        assert.equal(exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape'), undefined);
        vm.runInNewContext(compiled, { ...context, CVAT_AUTOFIT_REAL_BUILD_ENABLED: true });
        exports.setAutoFitPreference(canvas, true);
        const draw = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape');
        assert.equal(exports.mockAutoFitEnabled, false);
        assert.equal(draw.provider.kind, 'geometry-only');
        annotation.job.instance.labels = [{ id: 4, name: 'Pedestrian' }];
        const pedestrian = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape');
        assert.equal(pedestrian.isCurrent(), true);
        assert.equal(pedestrian.identity.vehicleType, undefined);
        assert.equal(draw.isCurrent(), false);
        annotation.job.instance.labels = [{ id: 4, name: 'Animal' }];
        assert.equal(exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape'), undefined);
        annotation.job.instance.labels = [{ id: 4, name: 'vehicles', attributes: [{ id: 143,
            name: 'type', inputType: 'select', mutable: false, values: ['car', 'truck', 'bus', 'van', 'other'],
            defaultValue: 'other' }] }];
        exports.setVehicleTypePreference(canvas, 2, 4, 'other');
        assert.equal(exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape', 'other'), undefined);
        for (const type of ['car', 'truck', 'bus', 'van']) {
            exports.setVehicleTypePreference(canvas, 2, 4, type);
            const vehicle = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape', type);
            assert.equal(vehicle.identity.typeAttributeId, 143);
            assert.equal(vehicle.isCurrent(), true);
            assert.equal(exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape').identity.vehicleType, type);
            exports.setVehicleTypePreference(canvas, 2, 4, 'other');
            assert.equal(vehicle.isCurrent(), false);
        }
        annotation.job.instance.labels = [{ id: 4, name: 'two-wheels', attributes: [{ id: 294,
            name: 'ride_state', inputType: 'checkbox', mutable: false, defaultValue: 'false' }] }];
        const wheels = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape');
        assert.equal(wheels.identity.rideState, false);
        assert.equal(wheels.identity.rideStateAttributeId, 294);
        assert.equal(wheels.isCurrent(), true);
        exports.setRideStatePreference(canvas, 2, 4, true);
        assert.equal(wheels.isCurrent(), false);
        assert.equal(exports.rideAnnotationAttribute(canvas, 2, annotation.job.instance.labels[0])[294], 'true');
        const rider = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape');
        assert.equal(rider.isCurrent(), true);
        annotation.job.instance.labels[0].attributes[0].id = 295;
        assert.equal(rider.isCurrent(), false);
        annotation.job.instance.labels = [{ id: 4, name: 'Car' }];
        assert.equal(draw.isCurrent(), true);
        annotation.job.instance = { ...job }; // A same-job metadata refresh preserves the session.
        assert.equal(draw.isCurrent(), true);
        assert.equal(exports.autoFitPreference(canvas), true);
        for (const changed of [{ ...job, id: 99 }, { ...job, taskId: 99 }]) {
            annotation.job.instance = changed;
            assert.equal(draw.isCurrent(), false);
        }
        annotation.job.instance = { ...job };
        annotation.player.frame.number = 99;
        assert.equal(draw.isCurrent(), false);
        annotation.player.frame.number = 3;
        for (const field of ['activeLabelID', 'activeObjectType', 'activeShapeType']) {
            const previous = annotation.drawing[field];
            annotation.drawing[field] = 'changed';
            assert.equal(draw.isCurrent(), false);
            annotation.drawing[field] = previous;
        }
        annotation.canvas.activeControl = 'cursor';
        assert.equal(draw.isCurrent(), false);
        annotation.canvas.activeControl = 'draw-cuboid';
        const unavailable = await draw.provider.fit(request, new AbortController().signal);
        assert.equal(unavailable.status, 'no-fit');
        assert.equal(unavailable.reason, 'unavailable');
        const op = new AutoFitOperation();
        assert.equal((await op.start(draw, seed)).hint, 'Auto Fit is unavailable. Try again later.');
        annotation.canvas.instance = new Canvas3d();
        assert.equal(draw.isCurrent(), false);
        assert.equal(exports.autoFitPreference(annotation.canvas.instance), false);
    }
});

test('webpack owns the mock gate independently of dotenv and forces it OFF for production', () => {
    const config = require('../../cvat-ui/webpack.config.js');
    const previousFlag = process.env.CVAT_AUTOFIT_MOCK;
    const previousRealFlag = process.env.CVAT_AUTOFIT_ENABLED;
    const previousServe = process.env.WEBPACK_SERVE;
    try {
        process.env.CVAT_AUTOFIT_MOCK = 'true';
        delete process.env.CVAT_AUTOFIT_ENABLED;
        delete process.env.WEBPACK_SERVE;
        for (const [mode, expected] of [['production', 'false'], ['development', 'true']]) {
            const definition = config({}, { mode }).plugins.find((plugin) => plugin.definitions?.CVAT_AUTOFIT_MOCK_BUILD_ENABLED);
            assert.equal(definition.definitions.CVAT_AUTOFIT_MOCK_BUILD_ENABLED, expected);
            assert.equal(definition.definitions['process.env.CVAT_AUTOFIT_MOCK'], undefined);
            assert.equal(definition.definitions.CVAT_AUTOFIT_REAL_BUILD_ENABLED, 'false');
        }
        process.env.CVAT_AUTOFIT_ENABLED = 'true';
        const enabled = config({}, { mode: 'development' }).plugins.find((plugin) =>
            plugin.definitions?.CVAT_AUTOFIT_REAL_BUILD_ENABLED);
        assert.equal(enabled.definitions.CVAT_AUTOFIT_REAL_BUILD_ENABLED, 'true');
    } finally {
        if (previousFlag === undefined) delete process.env.CVAT_AUTOFIT_MOCK;
        else process.env.CVAT_AUTOFIT_MOCK = previousFlag;
        if (previousRealFlag === undefined) delete process.env.CVAT_AUTOFIT_ENABLED;
        else process.env.CVAT_AUTOFIT_ENABLED = previousRealFlag;
        if (previousServe === undefined) delete process.env.WEBPACK_SERVE;
        else process.env.WEBPACK_SERVE = previousServe;
    }
});

test('all mock failure modes recover for retry; mock is deterministic and abortable', async () => {
    for (const mode of ['no-fit', 'reject', 'nan', 'infinity', 'negative',
        'wrong-frame', 'wrong-request', 'wrong-provider']) {
        const op = new AutoFitOperation();
        assert.equal((await op.start({ identity, isCurrent: () => true,
            provider: createMockAutoFitProvider(mode, 0) }, seed)).status, 'retry');
        assert.equal((await op.start({ identity, isCurrent: () => true,
            provider: createMockAutoFitProvider('success', 0) }, seed)).status, 'ok');
    }
    const provider = createMockAutoFitProvider('success', 0);
    const signal = new AbortController().signal;
    assert.deepEqual(await provider.fit(request, signal), await provider.fit(request, signal));
    const op = new AutoFitOperation();
    const slow = op.start({ identity, isCurrent: () => true, provider: createMockAutoFitProvider('timeout') }, seed);
    op.cancel();
    assert.equal((await slow).status, 'cancelled');
});

test('real provider forwards same identity without retries and cancels on abort or deadline', async () => {
    const source = fs.readFileSync(path.resolve(__dirname, '../../cvat-ui/src/utils/auto-fit-provider.ts'), 'utf8');
    const compiled = ts.transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const calls = [];
    let post = async () => ({ data: { ...result, provider: 'geometry-only' } });
    const axios = {
        post: (...args) => { calls.push(['POST', ...args]); return post(...args); },
        delete: (...args) => { calls.push(['DELETE', ...args]); return Promise.resolve({ status: 204 }); },
        isAxiosError: (error) => !!error.isAxiosError,
    };
    const exports = {};
    vm.runInNewContext(compiled, { exports, require(name) {
        if (name === 'axios') return { default: axios };
        if (name === 'cvat-store') return {
            getCVATStore: () => ({ getState: () => ({ organizations: { current: { slug: 'test-org' } } }) }),
        };
        assert.fail(`Unexpected provider import ${name}`);
    } });
    const provider = exports.createRealAutoFitProvider('geometry-only');
    assert.equal((await provider.fit(request, new AbortController().signal)).provider, 'geometry-only');
    assert.equal(calls[0][1], '/api/jobs/2/auto-fit');
    assert.equal(calls[0][2], request);
    assert.equal(calls[0][3]['axios-retry'].retries, 0);
    assert.equal(calls[0][3].params.org, 'test-org');
    const controller = new AbortController();
    post = (_url, _data, config) => new Promise((_resolve, reject) => {
        config.signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true });
    });
    const pending = provider.fit(request, controller.signal);
    controller.abort();
    await assert.rejects(pending);
    assert.equal(calls.at(-1)[0], 'DELETE');
    assert.equal(calls.at(-1)[1], '/api/jobs/2/auto-fit/fixture');
    post = async () => { throw { isAxiosError: true, response: { status: 504 } }; };
    assert.equal((await provider.fit(request, new AbortController().signal)).reason, 'timeout');
    assert.equal(calls.at(-1)[0], 'DELETE');
    post = async () => { throw { isAxiosError: true, response: { status: 503 } }; };
    assert.equal((await provider.fit(request, new AbortController().signal)).reason, 'unavailable');
});

test('actual point picker ignores template, returns vertex XYZ/index and rejects empty/out-of-viewport coordinates', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 10);
    camera.lookAt(0, 0, 0);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0.01, 0, 0, 2, 2, -100], 3));
    const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.05 }));
    scene.add(cloud);
    const template = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 4), new THREE.MeshBasicMaterial());
    template.name = 'drawTemplate';
    scene.add(template);
    const picked = pickAutoFitSeed(scene, camera, new THREE.Vector2(0, 0), 1000, 1000);
    assert.equal(picked.pointIndex, 0);
    assert.equal(picked.x, geometry.getAttribute('position').getX(0));
    assert.equal(picked.z, 0); // Not the ray intersection or cube surface.
    assert.equal(pickAutoFitSeed(scene, camera, new THREE.Vector2(0.8, 0.8), 1000, 1000), null);
    assert.equal(pickAutoFitSeed(scene, camera, new THREE.Vector2(2, 0), 1000, 1000), null);
    assert.equal(pickAutoFitSeed(scene, camera, new THREE.Vector2(NaN, 0), 1000, 1000), null);
    assert.equal(pickAutoFitSeed(scene, camera, new THREE.Vector2(0, 0), 0, 1000), null);
    cloud.visible = false;
    assert.equal(pickAutoFitSeed(scene, camera, new THREE.Vector2(0, 0), 1000, 1000), null);
});

test('estimated results require negotiation and honest geometry fallback provenance', () => {
    const estimated = { ...result, quality: 'estimated' };
    assert.throws(() => fitResultToPoints(estimated, request, 'mock'));
    assert.equal(fitResultToPoints(estimated, { ...request, allowEstimated: true }, 'mock').length, 16);
    assert.throws(() => fitResultToPoints({ ...result, quality: 'anything' }, request, 'mock'));
    const fallback = { ...estimated, provider: 'geometry-only' };
    assert.equal(fitResultToPoints(fallback, { ...request, allowEstimated: true }, 'snap').length, 16);
    assert.throws(() => fitResultToPoints({ ...fallback, quality: 'confident' },
        { ...request, allowEstimated: true }, 'snap'));
});

function loadView(parse) {
    const compiled = ts.transpileModule(fs.readFileSync(path.resolve(__dirname,
        '../src/typescript/canvas3dView.ts'), 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const exports = {};
    vm.runInNewContext(compiled, {
        exports, console, CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        window: { document: { getElementsByClassName: () => [] } },
        require(name) {
            if (name === 'three') return THREE;
            if (name.includes('PCDLoader')) return { PCDLoader: class { parse(data) { return parse(data); } } };
            if (name === './canvas3dModel') return require('../src/typescript/canvas3dModel.ts');
            if (name === './cuboid') return { CuboidModel: class {
                constructor() { this.perspective = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()); }
                setOrientationVisibility() {}
                setName(name) { this.perspective.name = name; }
            } };
            if (name === './autoFit') return { AutoFitOperation };
            if (name === './autoFitSeed') return { pickAutoFitSeed };
            return {};
        },
    });
    return exports.Canvas3dViewImpl;
}

function viewFixture(View) {
    const events = [];
    const scene = new THREE.Scene();
    const cloud = new THREE.Points(new THREE.BufferGeometry().setAttribute('position',
        new THREE.Float32BufferAttribute([0, 0, 0], 3)), new THREE.PointsMaterial({ size: .05 }));
    scene.add(cloud);
    const camera = new THREE.PerspectiveCamera(50, 1, .1, 1000);
    camera.position.set(0, 0, 10);
    camera.lookAt(0, 0, 0);
    const model = { data: { imageID: 3, isFrameUpdating: true,
        image: { imageData: { arrayBuffer: async () => new ArrayBuffer(1) } },
        configuration: { orientationVisibility: 'all' } },
        mode: Mode.DRAW, updateCanvasObjects() {}, unlockFrameUpdating() { this.data.isFrameUpdating = false; } };
    const view = Object.create(View.prototype);
    Object.assign(view, { model, controller: { mode: Mode.DRAW, imageIsDeleted: false, drawData: { enabled: true, autoFit: {} } },
        imageGeneration: 0, autoFitPickKey: '', autoFitPointerInside: true,
        autoFitOperation: new AutoFitOperation(), clearScene() {}, deactivateObject() {},
        onSceneImageLoaded() {}, views: { perspective: { scene, camera,
            renderer: { dispose() {}, domElement: { getBoundingClientRect: () => ({ width: 1000, height: 1000 }),
                dispatchEvent(event) { events.push(event); } } },
            rayCaster: { renderer: new THREE.Raycaster(), mouseVector: new THREE.Vector2(0, 0) } } } });
    return { view, model, events, cloud };
}

test('real canvas orchestration unlocks rejected PCD reads/parses and can load again', async () => {
    for (const failure of ['read', 'parse']) {
        let badParse = failure === 'parse';
        const View = loadView(() => {
            if (badParse) throw new Error('broken PCD');
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute([1, 2, 3], 3));
            return new THREE.Points(geometry, new THREE.PointsMaterial());
        });
        const { view, model, events } = viewFixture(View);
        if (failure === 'read') model.data.image.imageData.arrayBuffer = async () => { throw new Error('read failed'); };
        view.notify(model, UpdateReasons.IMAGE_CHANGED);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(model.data.isFrameUpdating, false);
        assert.equal(events.filter((event) => event.type === 'canvas.error').length, 1);
        assert.equal(events.filter((event) => event.type === 'canvas.setup').length, 0);
        badParse = false;
        model.data.isFrameUpdating = true;
        model.data.image.imageData.arrayBuffer = async () => new ArrayBuffer(1);
        view.notify(model, UpdateReasons.IMAGE_CHANGED);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(model.data.isFrameUpdating, false);
        assert.equal(events.filter((event) => event.type === 'canvas.setup').length, 1);
    }
});

test('auto fit preview is attached only by draw, moves to real point, hides on empty space and skips idle picks', () => {
    const View = loadView(() => assert.fail('Unexpected parse'));
    const { view, model, cloud } = viewFixture(View);
    assert.equal(cloud.getObjectByName('drawTemplate'), undefined);
    view.notify(model, UpdateReasons.DRAW);
    assert.equal(cloud.getObjectByName('drawTemplate'), view.cube.perspective);
    view.renderRayCaster(view.views.perspective);
    assert.equal(view.cube.perspective.visible, true);
    assert.equal(view.cube.perspective.position.z, 0);
    const savedRay = view.views.perspective.rayCaster.renderer;
    // The picker owns a separate raycaster; changing the key exposes whether idle work is skipped.
    const key = view.autoFitPickKey;
    view.renderRayCaster(view.views.perspective);
    assert.equal(view.autoFitPickKey, key);
    assert.equal(view.views.perspective.rayCaster.renderer, savedRay);
    view.views.perspective.rayCaster.mouseVector.set(.9, .9);
    view.renderRayCaster(view.views.perspective);
    assert.equal(view.cube.perspective.visible, false);
});

test('preparation stays separate from drawing, supplies scoped token and stops after frame change', async () => {
    const compiled = ts.transpileModule(fs.readFileSync(path.resolve(__dirname,
        '../../cvat-ui/src/utils/auto-fit-provider.ts'), 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const calls = [];
    let resolveOld;
    const axios = {
        post: async (url, data, config) => {
            calls.push({ url, data, config });
            if (url.endsWith('/prepare')) {
                if (data.frame === 4) return new Promise((resolve) => { resolveOld = resolve; });
                return { data: { frame: data.frame, frameToken: 'a'.repeat(32), state: 'ready' } };
            }
            return { data: { ...result, provider: 'snap' } };
        },
        delete: async () => ({}), isAxiosError: () => false,
    };
    const exports = {};
    vm.runInNewContext(compiled, { exports, AbortController, setTimeout, clearTimeout,
        require(name) {
            if (name === 'axios') return { default: axios };
            if (name === 'cvat-store') return {
                getCVATStore: () => ({ getState: () => ({ organizations: { current: null } }) }),
            };
            assert.fail(name);
        },
    });
    const canvas = {};
    const stop = exports.prepareRealAutoFitFrame(canvas, identity);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/jobs/2/auto-fit/prepare');
    const provider = exports.createRealAutoFitProvider('snap', canvas);
    await provider.fit(request, new AbortController().signal);
    assert.equal(calls.at(-1).data.frameToken, 'a'.repeat(32));
    const stopOld = exports.prepareRealAutoFitFrame(canvas, { ...identity, frame: 4 });
    stopOld();
    resolveOld({ data: { frame: 4, frameToken: 'b'.repeat(32), state: 'ready' } });
    await new Promise((resolve) => setImmediate(resolve));
    await provider.fit({ ...request, frame: 4 }, new AbortController().signal);
    assert.equal(calls.at(-1).data.frameToken, undefined);
    stop();
});

test('destroy during a delayed PCD read cannot publish setup or attach geometry', async () => {
    let resolveRead;
    let parseCalls = 0;
    const View = loadView(() => { parseCalls += 1; assert.fail('Stale read parsed'); });
    const { view, model, events } = viewFixture(View);
    model.data.image.imageData.arrayBuffer = () => new Promise((resolve) => { resolveRead = resolve; });
    view.notify(model, UpdateReasons.IMAGE_CHANGED);
    view.notify(model, UpdateReasons.DESTROY);
    resolveRead(new ArrayBuffer(1));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(parseCalls, 0);
    assert.equal(events.length, 0);
});


test('provenance is validated and captured with the accepted creation transaction', async () => {
    const metadata = { version: 1, request_id: 'fixture', frame: 3, class_key: 'pedestrian',
        quality: 'estimated', fit_mode: 'hybrid_prior', yaw_uncertain: true, prototype_key: 'pedestrian:adult',
        prior_version: 'pr0-dev', experimental: true };
    const estimated = { ...result, quality: 'estimated', autoFit: metadata };
    const estimatedRequest = { ...request, allowEstimated: true };
    assert.equal(fitResultToPoints(estimated, estimatedRequest, 'mock').length, 16);
    for (const override of [{ request_id: 'stale' }, { frame: 4 }, { quality: 'confident' },
        { class_key: 'bicycle' }, { fit_mode: 'guess' }, { yaw_uncertain: 'true' }, { experimental: undefined },
        { prototype_key: 42 }, { prior_version: {} }]) {
        assert.throws(() => fitResultToPoints({ ...estimated, autoFit: { ...metadata, ...override } },
            estimatedRequest, 'mock'));
    }
    const op = new AutoFitOperation();
    const outcome = await op.start({ identity, isCurrent: () => true, allowEstimated: true,
        provider: { kind: 'mock', fit: async (captured) => ({ ...estimated, requestId: captured.requestId,
            autoFit: { ...metadata, request_id: captured.requestId } }) } }, seed);
    assert.equal(outcome.status, 'ok');
    assert.equal(outcome.request.coordinateSystem, 'lidar');
    assert.equal(outcome.request.units, 'metres');
    assert.equal(outcome.autoFit.request_id, outcome.request.requestId);
    assert.equal(outcome.autoFit.yaw_uncertain, true);
    assert.equal(outcome.autoFit.experimental, true);
});


test('unsafe fit reports missing reliable prior without manufacturing an annotation', async () => {
    const op = new AutoFitOperation();
    const outcome = await op.start({ identity, isCurrent: () => true, allowEstimated: true,
        provider: { kind: 'snap', fit: async (captured) => ({ status: 'no-fit', reason: 'unsafe-fit',
            requestId: captured.requestId, frame: captured.frame, diagnostics: { diagnosticFlags: ['prior-unavailable'] } }) } }, seed);
    assert.equal(outcome.status, 'retry');
    assert.match(outcome.hint, /reliable size prior is unavailable/);
    assert.equal(op.inFlight, false);
});
