import { createHash, randomUUID } from 'node:crypto';
import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { gunzipSync } from 'node:zlib';

export const labels = [
    { name: 'vehicles', type: 'cuboid', attributes: [{ name: 'type', mutable: false,
        input_type: 'select', default_value: 'other', values: ['car', 'truck', 'bus', 'van', 'other'] }] },
    { name: 'pedestrian', type: 'cuboid', attributes: ['age', 'pose'].map((name) => ({
        name, mutable: false, input_type: 'text', default_value: '', values: [],
    })) },
    { name: 'two-wheels', type: 'cuboid', attributes: [{ name: 'ride_state', mutable: false,
        input_type: 'checkbox', default_value: 'false', values: ['false'] }] },
    { name: 'Animal', type: 'cuboid', attributes: [] },
    { name: 'Obstacle', type: 'cuboid', attributes: [] },
];

export const sha256 = (buffer) => createHash('sha256').update(buffer).digest('hex');
export const runName = (project) => `autofit-pr0-${project}-${randomUUID()}`;

export function assertOwnedTask(task, owned) {
    if (!Number.isInteger(owned.id) || task.id !== owned.id || task.name !== owned.name ||
        !/^autofit-pr0-(mocked|nuscenes)-[0-9a-f-]{36}$/.test(owned.name)) {
        throw new Error('Refusing mutation of a task outside this run namespace');
    }
}

export function resolveSchema(taskLabels) {
    const vehicle = taskLabels.find((label) => label.name === 'vehicles');
    const type = vehicle?.attributes.find((attribute) => attribute.name === 'type');
    if (!Number.isInteger(vehicle?.id) || !Number.isInteger(type?.id) || type.mutable !== false ||
        type.input_type !== 'select' || JSON.stringify(type.values) !== JSON.stringify(labels[0].attributes[0].values)) {
        throw new Error('Runtime task taxonomy does not match the locked vehicles/type contract');
    }
    const pedestrian = taskLabels.find((label) => label.name === 'pedestrian');
    const twoWheels = taskLabels.find((label) => label.name === 'two-wheels');
    const rideState = twoWheels?.attributes.find((attribute) => attribute.name === 'ride_state');
    if (!Number.isInteger(pedestrian?.id) || !Number.isInteger(twoWheels?.id) ||
        !Number.isInteger(rideState?.id) || rideState.input_type !== 'checkbox' || rideState.mutable !== false) {
        throw new Error('Runtime task taxonomy does not match pedestrian/two-wheels contract');
    }
    return { vehicleLabelId: vehicle.id, typeAttributeId: type.id,
        pedestrianLabelId: pedestrian.id, twoWheelsLabelId: twoWheels.id, rideStateAttributeId: rideState.id };
}

export function allowLocalSkip(env = process.env) {
    return env.AUTOFIT_ALLOW_LOCAL_SKIP === '1' && !env.CI && env.AUTOFIT_ACCEPTANCE !== '1';
}

export function prerequisites(real, env = process.env) {
    const required = ['CVAT_USERNAME', 'CVAT_PASSWORD'];
    if (real) required.push('AUTOFIT_MANIFEST', 'AUTOFIT_MANIFEST_SHA256', 'AUTOFIT_GPU_HEALTH_URL');
    return required.filter((key) => !env[key]).map((key) => `Missing ${key}`);
}

export function toPCD(points) {
    const dimensions = points[0]?.length;
    if (![3, 4].includes(dimensions) || points.some((point) =>
        point.length !== dimensions || !point.every(Number.isFinite))) {
        throw new Error('PCD fixture needs uniform finite XYZ or XYZI points; never silently reindex a seed');
    }
    const fields = dimensions === 4 ? 'x y z intensity' : 'x y z';
    const header = `# .PCD v0.7\nVERSION 0.7\nFIELDS ${fields}\nSIZE ${Array(dimensions).fill(4).join(' ')}\nTYPE ${Array(dimensions).fill('F').join(' ')}\nCOUNT ${Array(dimensions).fill(1).join(' ')}\nWIDTH ${points.length}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS ${points.length}\nDATA binary\n`;
    const stride = dimensions * 4;
    const data = Buffer.alloc(points.length * stride);
    points.forEach((point, index) => point.forEach((value, axis) => data.writeFloatLE(value, index * stride + axis * 4)));
    return Buffer.concat([Buffer.from(header), data]);
}

export function syntheticPCD() {
    const points = [];
    for (let x = -2; x <= 2; x += 0.2) {
        for (let y = -1; y <= 1; y += 0.2) points.push([x, y, 1]);
    }
    return toPCD(points);
}

export async function loadRealFixture(env = process.env) {
    const bytes = await readFile(env.AUTOFIT_MANIFEST);
    if (!/^[0-9a-f]{64}$/.test(env.AUTOFIT_MANIFEST_SHA256 || '') || sha256(bytes) !== env.AUTOFIT_MANIFEST_SHA256) {
        throw new Error('Manifest SHA256 differs from the explicitly locked fixture');
    }
    const manifest = JSON.parse((bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes).toString());
    // The default remains the locked Car regression fixture. Explicit cases are small development
    // smoke tests, including sparse/adjacent failures; selecting one does not imply held-out acceptance.
    const mapped = (item) => (item.target_label === 'vehicles' && ['car', 'truck', 'bus'].includes(item.vehicles_type)) ||
        ['pedestrian', 'two-wheels'].includes(item.target_label);
    const candidates = manifest.cases.filter((item) => ['dev', 'held_out'].includes(item.split) &&
        mapped(item) && item.seeds?.length);
    candidates.sort((a, b) => a.annotation_token.localeCompare(b.annotation_token));
    const selected = env.AUTOFIT_CASE_ID ? candidates.find((item) => item.case_id === env.AUTOFIT_CASE_ID) :
        candidates.find((item) => item.split === 'held_out' && item.target_label === 'vehicles' && item.vehicles_type === 'car');
    if (!selected) throw new Error('Locked manifest has no requested seedable mapped dev/held-out case or default held-out Car');
    const frame = manifest.frames[selected.sample_token];
    if (!frame || frame.split !== selected.split || frame.point_dimensions !== 5) throw new Error('Invalid LiDAR frame contract');
    // Frozen PR0 source mappings are semantic, not task numeric IDs. Van must remain an explicit
    // request override of a Car–Van-pool fixture; never relabel vehicle.car as independent Van GT.
    const category = selected.source_category;
    if (category !== undefined) {
        const validSource = (selected.target_label === 'vehicles' &&
            ((selected.vehicles_type === 'car' && category === 'vehicle.car') ||
             (selected.vehicles_type === 'truck' && category === 'vehicle.truck') ||
             (selected.vehicles_type === 'bus' && category.startsWith('vehicle.bus.')))) ||
            (selected.target_label === 'pedestrian' && category.startsWith('human.pedestrian.')) ||
            (selected.target_label === 'two-wheels' && ['vehicle.bicycle', 'vehicle.motorcycle'].includes(category));
        if (!validSource) throw new Error('Fixture source category conflicts with the locked public mapping');
    }
    if (frame.lidar_to_global !== undefined) {
        const matrix = frame.lidar_to_global;
        if (!Array.isArray(matrix) || matrix.length !== 4 || matrix.some((row) =>
            !Array.isArray(row) || row.length !== 4 || !row.every(Number.isFinite)) ||
            matrix[3].some((value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-6)) {
            throw new Error('Invalid frozen LiDAR calibration');
        }
        const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
        const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if (Math.abs(determinant - 1) > 1e-5) throw new Error('Invalid frozen LiDAR calibration rotation');
    }
    const root = await realpath(env.NUSCENES_DATAROOT || path.join(os.homedir(), 'Datasets/nuScenes'));
    const cloudPath = await realpath(path.resolve(root, frame.lidar_path));
    if (!cloudPath.startsWith(`${root}${path.sep}`)) throw new Error('Manifest LiDAR path escapes dataroot');
    const cloud = await readFile(cloudPath);
    if (sha256(cloud) !== frame.sha256 || cloud.length % 20) throw new Error('LiDAR file hash/stride mismatch');
    const points = Array.from({ length: cloud.length / 20 }, (_, index) => (
        [0, 1, 2, 3].map((axis) => cloud.readFloatLE(index * 20 + axis * 4))
    ));
    const seed = selected.seeds[0];
    if (!Array.isArray(seed.point) || seed.point.length !== 3 || !seed.point.every(Number.isFinite) ||
        !Number.isInteger(seed.pointIndex) || !points[seed.pointIndex] ||
        points[seed.pointIndex].slice(0, 3).some((value, axis) => Math.abs(value - seed.point[axis]) > 1e-5)) {
        throw new Error('Manifest seed does not identify the original LiDAR vertex');
    }
    // SNAP consumes intensity. Drop only ring, preserving every original XYZI float32 and index.
    const pcd = toPCD(points);
    return { files: [{ name: '000000.pcd', buffer: pcd }], selected,
        manifestHash: sha256(bytes), cloudHash: frame.sha256, pcdHash: sha256(pcd), inputFields: 'xyzi' };
}
