import { test, expect } from '../support/fixtures.mjs';
import { openJob, startDrawing, confirmPoint, objectsSelector } from '../support/ui.mjs';

test('PW-14 @real-nuscenes @gpu Car fixture reaches real SNAP and commits once', async ({ page, isolated }, info) => {
    test.setTimeout(240_000);
    const preparation = { taskId: isolated.id, jobId: isolated.jobId, frame: 0 };
    const prepared = await isolated.api.post(`/api/jobs/${isolated.jobId}/auto-fit/prepare`, { data: preparation });
    expect(prepared.ok()).toBe(true);
    const { frameToken } = await prepared.json();
    await expect.poll(async () => {
        const response = await isolated.api.get(`/api/jobs/${isolated.jobId}/auto-fit/prepare`, {
            params: { ...preparation, frameToken },
        });
        expect(response.ok()).toBe(true);
        return (await response.json()).state;
    }, { timeout: 120_000, intervals: [500, 1000, 2000] }).toBe('ready');
    await openJob(page, isolated);
    await startDrawing(page);
    const fit = page.waitForResponse((response) => new URL(response.url()).pathname ===
        `/api/jobs/${isolated.jobId}/auto-fit` && response.request().method() === 'POST');
    await confirmPoint(page, isolated.selected.seeds[0]);
    const response = await fit;
    const result = await response.json();
    const payload = response.request().postDataJSON();
    await info.attach('real-autofit-response', { body: JSON.stringify({ payload, result,
        requestTiming: response.request().timing() }, null, 2), contentType: 'application/json' });
    expect(response.ok()).toBe(true);
    expect(payload).toMatchObject({ vehicleType: 'car', labelId: isolated.vehicleLabelId,
        typeAttributeId: isolated.typeAttributeId });
    expect(payload.seed.pointIndex).toBe(isolated.selected.seeds[0].pointIndex);
    expect(result).toMatchObject({ status: 'ok', provider: 'snap', requestId: payload.requestId, frame: 0 });
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    // Full accuracy/latency gates belong to the locked evaluator; one browser smoke is not a baseline.
});
