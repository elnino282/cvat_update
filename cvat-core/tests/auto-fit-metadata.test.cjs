// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');

const compile = (source) => ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
}).outputText;
require.extensions['.ts'] = (module, filename) => module._compile(compile(fs.readFileSync(filename, 'utf8')), filename);

const { default: Collection } = require('../src/annotations-collection.ts');
const { default: ObjectState } = require('../src/object-state.ts');
const { default: History } = require('../src/annotations-history.ts');
const { Label } = require('../src/labels.ts');
const { copyAutoFitMetadata } = require('../src/auto-fit-metadata.ts');
const metadata = {
    version: 1, request_id: 'test-fit', frame: 0, class_key: 'pedestrian',
    quality: 'estimated', fit_mode: 'hybrid_prior', yaw_uncertain: true,
    prototype_key: 'pedestrian-adult', prior_version: 'pr0-dev', experimental: true,
};

function fixture() {
    const label = new Label({ id: 7, name: 'pedestrian', type: 'cuboid', color: '#ff0000', attributes: [] });
    const history = new History();
    const collection = new Collection({ labels: [label], history, stopFrame: 2, dimension: '3d',
        jobType: 'annotation', framesInfo: {
            0: { width: 100, height: 100 }, 1: { width: 100, height: 100 },
            2: { width: 100, height: 100 }, isFrameDeleted: () => false,
        } });
    return { label, history, collection };
}

function createState(label, objectType, autoFit = metadata) {
    return new ObjectState({ objectType, label, frame: 0, shapeType: 'cuboid', autoFit,
        points: [1, 2, 1, 0, 0, 0, 0.6, 0.6, 1.8, 0, 0, 0, 0, 0, 0, 0] });
}

test('metadata is bounded, complete JSON provenance; legacy empty is accepted', () => {
    assert.deepEqual(copyAutoFitMetadata(), {});
    assert.deepEqual(copyAutoFitMetadata(metadata), metadata);
    for (const invalid of [null, [], 'x', { ...metadata, points: [] }, { ...metadata, frame: NaN },
        { ...metadata, frame: -1 }, { ...metadata, version: true }, { ...metadata, quality: 'ok' },
        { ...metadata, experimental: 1 }, { ...metadata, prototype_key: {} },
        { ...metadata, request_id: 'x'.repeat(129) }, { ...metadata, request_id: 'a\nb' }]) {
        assert.throws(() => copyAutoFitMetadata(invalid));
    }
    assert.throws(() => copyAutoFitMetadata({ version: 1 }));
});

for (const objectType of ['shape', 'track']) {
    test(`${objectType}: actual collection put, undo/redo, edit, copy and save/reload retain provenance`, async () => {
        const { collection, history, label } = fixture();
        const original = createState(label, objectType);
        const exposed = original.autoFit;
        exposed.quality = 'confident';
        assert.equal(original.autoFit.quality, 'estimated');
        collection.put([original]);
        assert.equal(collection.get(0, false, []).length, 1);
        await history.undo(1);
        assert.equal(collection.get(0, false, []).length, 0);
        await history.redo(1);
        let [state] = collection.get(0, false, []);
        assert.deepEqual(state.autoFit, metadata);
        const before = [...state.points];
        state.points = state.points.map((v, index) => index === 0 ? v + 0.2 : v);
        state = await state.save();
        assert.deepEqual(state.autoFit, metadata);
        await history.undo(1);
        assert.deepEqual(collection.get(0, false, [])[0].points, before);
        assert.deepEqual(collection.get(0, false, [])[0].autoFit, metadata);
        await history.redo(1);
        const copied = new ObjectState({ ...state.serialize(), clientID: null, serverID: null });
        assert.deepEqual(copied.autoFit, metadata);
        collection.put([copied]);
        const exported = collection.export();
        assert.equal(exported[`${objectType}s`].length, 2);
        assert.deepEqual(exported[`${objectType}s`][0].auto_fit, metadata);
        assert.equal('autoFit' in exported[`${objectType}s`][0], false);
        const reloaded = fixture().collection;
        reloaded.import(JSON.parse(JSON.stringify(exported)));
        assert.deepEqual(reloaded.get(0, false, [])[0].autoFit, metadata);
        if (objectType === 'track') {
            assert.deepEqual(reloaded.get(1, false, [])[0].autoFit, metadata);
            assert.equal(reloaded.get(1, false, [])[0].autoFit.frame, 0);
            assert.equal('auto_fit' in exported.tracks[0].shapes[0], false);
        }
    });
    test(`${objectType}: old annotations default to empty provenance`, () => {
        const { collection, label } = fixture();
        collection.put([createState(label, objectType, {})]);
        const exported = collection.export();
        delete exported[`${objectType}s`][0].auto_fit;
        const reloaded = fixture().collection;
        reloaded.import(exported);
        assert.deepEqual(reloaded.get(0, false, [])[0].autoFit, {});
    });
}

// Execute production saver comparisons/diff, with only transport/session replaced.
function loadSaver() {
    const filename = path.resolve(__dirname, '../src/annotations-saver.ts');
    const loaded = new Module(filename, module);
    loaded.filename = filename;
    loaded.paths = Module._nodeModulePaths(path.dirname(filename));
    const originalRequire = loaded.require.bind(loaded);
    loaded.require = (name) => {
        if (name === './server-proxy') return {};
        if (name === './session') return { Task: class Task {} };
        return originalRequire(name);
    };
    loaded._compile(compile(`${fs.readFileSync(filename, 'utf8')}
        export { isTheSameShape, isTheSameTrack };`), filename);
    return loaded.exports;
}
const { default: Saver, isTheSameShape, isTheSameTrack } = loadSaver();

for (const objectType of ['shape', 'track']) {
    test(`${objectType}: save detects metadata-only changes and 504 recovery does not lose provenance`, () => {
        const { collection, label } = fixture();
        collection.put([createState(label, objectType)]);
        const initial = collection.export();
        initial[`${objectType}s`][0].id = 71;
        const saver = new Saver({ export: () => structuredClone(initial) }, { id: 1 });
        const changed = structuredClone(initial);
        changed[`${objectType}s`][0].auto_fit.quality = 'confident';
        assert.equal(saver._split(changed).updated[`${objectType}s`].length, 1);
        const match = objectType === 'shape' ? isTheSameShape : isTheSameTrack;
        const original = initial[`${objectType}s`][0];
        assert.equal(match(original, structuredClone(original)), true);
        assert.equal(match(original, changed[`${objectType}s`][0]), false);
        assert.equal(match(original, { ...original, auto_fit: {} }), false);
        const missing = { ...original };
        delete missing.auto_fit;
        assert.equal(match(missing, { ...original, auto_fit: {} }), true);
        assert.equal(match(original, { ...original,
            auto_fit: Object.fromEntries(Object.entries(metadata).reverse()) }), true);
    });
}
