import { test as base, expect, request } from 'playwright/test';
import { writeFile } from 'node:fs/promises';
import { labels, runName, assertOwnedTask, resolveSchema, prerequisites, allowLocalSkip,
    syntheticPCD, loadRealFixture } from './contract.mjs';

async function json(response, description) {
    if (!response.ok()) throw new Error(`${description}: HTTP ${response.status()}`);
    return response.json();
}

export async function assertGPU(api) {
    const health = await json(await api.get(process.env.AUTOFIT_GPU_HEALTH_URL), 'GPU health');
    if (health.status !== 'ready' || health.backend !== 'snap' || health.snapAvailable !== true ||
        !health.provenance?.device || !health.provenance?.cuda) {
        throw new Error('Real suite requires ready SNAP with recorded CUDA/device provenance');
    }
    return health;
}

async function createTask(api, real, fixture, info) {
    const created = await json(await api.post('/api/tasks', {
        data: { name: runName(real ? 'nuscenes' : 'mocked'), labels, segment_size: 2 },
    }), 'Create isolated task');
    const owned = { id: created.id, name: created.name };
    // Record ownership immediately so a failed upload is still diagnosable and cleaned up.
    await writeFile(info.outputPath('task-ownership.json'), JSON.stringify(owned, null, 2));
    try {
        const url = `/api/tasks/${owned.id}/data`;
        const start = await api.post(url, { headers: { 'Upload-Start': 'true' }, data: {} });
        if (!start.ok()) throw new Error(`Upload start: HTTP ${start.status()}`);
        const multipart = { image_quality: '100', use_zip_chunks: 'true', sorting_method: 'lexicographical' };
        fixture.files.forEach((file, index) => {
            multipart[`client_files[${index}]`] = { ...file, mimeType: 'application/octet-stream' };
        });
        const upload = await api.post(url, { headers: { 'Upload-Multiple': 'true' }, multipart });
        if (!upload.ok()) throw new Error(`Upload cloud: HTTP ${upload.status()}`);
        const finish = await json(await api.post(url, { headers: { 'Upload-Finish': 'true' },
            data: { image_quality: 100, use_zip_chunks: true, sorting_method: 'lexicographical' } }), 'Finish upload');
        if (!finish.rq_id) throw new Error('Task upload did not return rq_id');
        await expect.poll(async () => {
            const state = await json(await api.get(`/api/requests/${encodeURIComponent(finish.rq_id)}`), 'Poll task upload');
            if (state.status === 'failed') throw new Error(`Task import failed: ${state.message}`);
            return state.status;
        }, { timeout: 180_000, intervals: [500, 1000, 2000] }).toBe('finished');
        const jobs = await json(await api.get(`/api/jobs?task_id=${owned.id}`), 'Resolve fixture jobs');
        const job = jobs.results.find((item) => item.task_id === owned.id && item.type === 'annotation');
        if (!job || !Number.isInteger(job.id)) throw new Error('Fixture has no annotation job');
        const schema = await json(await api.get(`/api/labels?task_id=${owned.id}&page_size=100`), 'Resolve runtime labels');
        return { ...owned, jobId: job.id, ...resolveSchema(schema.results), ...fixture };
    } catch (error) {
        await destroyTask(api, owned);
        throw error;
    }
}

async function destroyTask(api, owned) {
    const response = await api.get(`/api/tasks/${owned.id}`);
    if (response.status() === 404) return;
    assertOwnedTask(await json(response, 'Verify task ownership before teardown'), owned);
    const deleted = await api.delete(`/api/tasks/${owned.id}`);
    if (!deleted.ok()) throw new Error(`Fixture cleanup failed: HTTP ${deleted.status()}; task ${owned.id}`);
}

export const test = base.extend({
    // Per-test fixture gives annotation isolation without deleting annotations from any existing task.
    isolated: [async ({}, use, info) => {
        const real = info.project.name === 'autofit-real-nuscenes';
        const missing = prerequisites(real);
        if (missing.length) {
            if (allowLocalSkip()) test.skip(true, missing.join('; '));
            throw new Error(missing.join('; '));
        }
        let fixture;
        let health;
        // Only absent/unavailable prerequisites may be locally skipped; assertions never become skips.
        const preflight = await request.newContext({ timeout: 10_000 });
        try {
            if (real) {
                fixture = await loadRealFixture();
                health = await assertGPU(preflight);
            } else {
                fixture = { files: ['000000.pcd', '000001.pcd'].map((name) => ({ name, buffer: syntheticPCD() })) };
            }
        } catch (error) {
            if (allowLocalSkip()) test.skip(true, `Fixture/GPU prerequisite: ${error.message}`);
            throw error;
        } finally {
            await preflight.dispose();
        }
        const api = await request.newContext({ baseURL: info.project.use.baseURL, timeout: 30_000 });
        let task;
        try {
            await json(await api.post('/api/auth/login', { data: {
                username: process.env.CVAT_USERNAME, password: process.env.CVAT_PASSWORD,
            } }), 'CVAT API login');
            // Session authentication requires CSRF on API writes after login.
            const state = await api.storageState();
            const csrf = state.cookies.find((cookie) => cookie.name === 'csrftoken')?.value;
            if (csrf) await api.dispose();
            const session = csrf ? await request.newContext({ baseURL: info.project.use.baseURL,
                storageState: state, extraHTTPHeaders: { 'X-CSRFToken': csrf }, timeout: 30_000 }) : api;
            try {
                task = await createTask(session, real, fixture, info);
                const evidence = { taskId: task.id, jobId: task.jobId, taskName: task.name,
                    vehicleLabelId: task.vehicleLabelId, typeAttributeId: task.typeAttributeId,
                    manifestHash: task.manifestHash, cloudHash: task.cloudHash,
                    caseId: task.selected?.case_id, annotationToken: task.selected?.annotation_token, health };
                await info.attach('fixture-evidence', { body: JSON.stringify(evidence, null, 2), contentType: 'application/json' });
                await use({ ...task, api: session, storageState: await session.storageState() });
            } finally {
                if (task) await destroyTask(session, task);
                if (session !== api) await session.dispose();
            }
        } finally {
            await api.dispose();
        }
    }, { timeout: 240_000 }],
    storageState: async ({ isolated }, use) => use(isolated.storageState),
});

export { expect };
