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

test('production and default development gate never construct a mock or read fixture overrides', async () => {
    const source = fs.readFileSync(path.resolve(__dirname, '../../cvat-ui/src/utils/auto-fit.ts'), 'utf8');
    const compiled = ts.transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    for (const [NODE_ENV, CVAT_AUTOFIT_MOCK] of [['production', 'true'], ['development', 'false']]) {
        class Canvas3d {}
        const canvas = new Canvas3d();
        const job = { id: 2, taskId: 1, dimension: '3d' };
        const annotation = {
            job: { instance: job }, player: { frame: { number: 3, fetching: false } },
            drawing: { activeLabelID: 4, activeObjectType: 'shape', activeShapeType: 'cuboid' },
            canvas: { instance: canvas, activeControl: 'draw-cuboid' },
        };
        const exports = {};
        vm.runInNewContext(compiled, {
            exports, process: { env: { NODE_ENV, CVAT_AUTOFIT_MOCK } },
            CVAT_AUTOFIT_MOCK_BUILD_ENABLED: CVAT_AUTOFIT_MOCK === 'true',
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
                assert.fail(`Unexpected import ${name}`);
            },
        });
        exports.setAutoFitPreference(canvas, true);
        const draw = exports.autoFitDrawData(canvas, 'cuboid', 4, 'shape');
        assert.equal(exports.mockAutoFitEnabled, false);
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
    const previousServe = process.env.WEBPACK_SERVE;
    try {
        process.env.CVAT_AUTOFIT_MOCK = 'true';
        delete process.env.WEBPACK_SERVE;
        for (const [mode, expected] of [['production', 'false'], ['development', 'true']]) {
            const definition = config({}, { mode }).plugins.find((plugin) => plugin.definitions?.CVAT_AUTOFIT_MOCK_BUILD_ENABLED);
            assert.equal(definition.definitions.CVAT_AUTOFIT_MOCK_BUILD_ENABLED, expected);
            assert.equal(definition.definitions['process.env.CVAT_AUTOFIT_MOCK'], undefined);
        }
    } finally {
        if (previousFlag === undefined) delete process.env.CVAT_AUTOFIT_MOCK;
        else process.env.CVAT_AUTOFIT_MOCK = previousFlag;
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
