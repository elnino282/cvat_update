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
    return { vehicleLabelId: vehicle.id, typeAttributeId: type.id };
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
    if (!points.length || points.some((point) => point.length < 3 || !point.slice(0, 3).every(Number.isFinite))) {
        throw new Error('PCD fixture needs finite XYZ points; never silently reindex a seed');
    }
    const header = `# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH ${points.length}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS ${points.length}\nDATA binary\n`;
    const data = Buffer.alloc(points.length * 12);
    points.forEach((point, index) => point.slice(0, 3).forEach((value, axis) => data.writeFloatLE(value, index * 12 + axis * 4)));
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
    const candidates = manifest.cases.filter((item) => item.split === 'held_out' &&
        item.target_label === 'vehicles' && item.vehicles_type === 'car' && item.seeds?.length);
    candidates.sort((a, b) => a.annotation_token.localeCompare(b.annotation_token));
    const selected = env.AUTOFIT_CASE_ID ? candidates.find((item) => item.case_id === env.AUTOFIT_CASE_ID) : candidates[0];
    if (!selected) throw new Error('Locked manifest has no requested held-out Car with point support');
    const frame = manifest.frames[selected.sample_token];
    if (!frame || frame.split !== 'held_out' || frame.point_dimensions !== 5) throw new Error('Invalid LiDAR frame contract');
    const root = await realpath(env.NUSCENES_DATAROOT || path.join(os.homedir(), 'Datasets/nuScenes'));
    const cloudPath = await realpath(path.resolve(root, frame.lidar_path));
    if (!cloudPath.startsWith(`${root}${path.sep}`)) throw new Error('Manifest LiDAR path escapes dataroot');
    const cloud = await readFile(cloudPath);
    if (sha256(cloud) !== frame.sha256 || cloud.length % 20) throw new Error('LiDAR file hash/stride mismatch');
    const points = Array.from({ length: cloud.length / 20 }, (_, index) => (
        [0, 1, 2].map((axis) => cloud.readFloatLE(index * 20 + axis * 4))
    ));
    const seed = selected.seeds[0];
    if (!Array.isArray(seed.point) || seed.point.length !== 3 || !seed.point.every(Number.isFinite) ||
        !Number.isInteger(seed.pointIndex) || !points[seed.pointIndex] ||
        points[seed.pointIndex].some((value, axis) => Math.abs(value - seed.point[axis]) > 1e-5)) {
        throw new Error('Manifest seed does not identify the original LiDAR vertex');
    }
    return { files: [{ name: '000000.pcd', buffer: toPCD(points) }], selected,
        manifestHash: sha256(bytes), cloudHash: frame.sha256 };
}
