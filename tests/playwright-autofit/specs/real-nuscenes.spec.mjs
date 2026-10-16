import { writeFile } from 'node:fs/promises';
import { test, expect } from '../support/fixtures.mjs';
import { openJob, startDrawing, confirmPoint, geometry, saveReload, pickedSeedEvidence, objectsSelector, canvasSelector } from '../support/ui.mjs';

test('PW-14/15/16/17 @real-nuscenes @gpu selected fixture uses real SNAP and preserves lifecycle', async ({ page, isolated }, info) => {
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
    const label = isolated.selected.target_label;
    // Van is an explicit interaction test of the shared Car–Van pool; it has no independent GT label here.
    const vehicleType = process.env.AUTOFIT_VEHICLE_TYPE || isolated.selected.vehicles_type;
    const rideState = process.env.AUTOFIT_RIDE_STATE === undefined ? Boolean(isolated.selected.ride_state) :
        process.env.AUTOFIT_RIDE_STATE === 'true';
    const mode = process.env.AUTOFIT_OBJECT_MODE || 'Shape';
    const expected = process.env.AUTOFIT_EXPECT_STATUS || 'ok';
    expect(['Shape', 'Track']).toContain(mode);
    expect(['ok', 'no-fit']).toContain(expected);
    await openJob(page, isolated);
    await startDrawing(page, mode, vehicleType, true, { label, rideState });
    const fit = page.waitForResponse((response) => new URL(response.url()).pathname ===
        `/api/jobs/${isolated.jobId}/auto-fit` && response.request().method() === 'POST');
    await confirmPoint(page, { ...isolated.selected.seeds[0], cameraView: process.env.AUTOFIT_SEED_VIEW });
    const response = await fit;
    const result = await response.json();
    const payload = response.request().postDataJSON();
    const responseEvidence = { payload, result, caseId: isolated.selected.case_id,
        cameraView: process.env.AUTOFIT_SEED_VIEW || 'lidar-origin',
        inputFields: isolated.inputFields, sourceCloudHash: isolated.cloudHash, uploadedPCDHash: isolated.pcdHash,
        requestTiming: response.request().timing(), independentVanGroundTruth: false };
    await writeFile(info.outputPath('real-autofit-response.json'), JSON.stringify(responseEvidence, null, 2));
    await info.attach('real-autofit-response', { body: JSON.stringify(responseEvidence, null, 2), contentType: 'application/json' });
    expect(response.ok()).toBe(true);
    expect(payload).toMatchObject({ objectType: mode.toLowerCase(), coordinateSystem: 'lidar', units: 'metres' });
    if (label === 'vehicles') expect(payload).toMatchObject({ vehicleType, labelId: isolated.vehicleLabelId,
        typeAttributeId: isolated.typeAttributeId });
    if (label === 'pedestrian') expect(payload.labelId).toBe(isolated.pedestrianLabelId);
    if (label === 'two-wheels') expect(payload).toMatchObject({ labelId: isolated.twoWheelsLabelId,
        rideState, rideStateAttributeId: isolated.rideStateAttributeId });
    const seedEvidence = await pickedSeedEvidence(page, payload.seed, isolated.selected);
    await writeFile(info.outputPath('picked-source-vertex.json'), JSON.stringify(seedEvidence, null, 2));
    await info.attach('picked-source-vertex', { body: JSON.stringify(seedEvidence, null, 2), contentType: 'application/json' });
    expect(result).toMatchObject({ status: expected, requestId: payload.requestId, frame: 0 });
    if (expected === 'no-fit') {
        const safeReasons = ['unsafe-fit', 'no-seed-support'];
        if (label === 'vehicles' && vehicleType === 'car') safeReasons.push('ambiguous', 'insufficient-evidence');
        expect(safeReasons).toContain(result.reason);
        if (process.env.AUTOFIT_EXPECT_REASON) expect(result.reason).toBe(process.env.AUTOFIT_EXPECT_REASON);
        await expect(page.locator(objectsSelector)).toHaveCount(0);
        await expect(page.getByTestId('autofit-status')).toBeVisible();
        if (result.reason === 'unsafe-fit') await expect(page.getByTestId('autofit-status')).toContainText('No safe fit');
        const retainedSeed = await page.locator(canvasSelector).evaluate((canvas) => {
            const template = canvas.scene.getObjectByName('drawTemplate');
            return template ? { position: template.position.toArray(), visible: template.visible } : null;
        });
        expect(retainedSeed, 'A rejected fit retains the drawing seed for retry/manual use').not.toBeNull();
        await writeFile(info.outputPath('no-fit-lifecycle.json'), JSON.stringify({ caseId: isolated.selected.case_id,
            annotationCount: 0, reason: result.reason, retainedSeed }, null, 2));
        return;
    }
    if (result.provider === 'geometry-only') {
        expect(result).toMatchObject({ quality: 'estimated', diagnostics: { fallbackFrom: 'snap' } });
    } else expect(result.provider).toBe('snap');
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    const before = await geometry(page);
    if (label === 'vehicles') expect(before[0].attributes[String(isolated.typeAttributeId)]).toBe(vehicleType);
    if (label === 'two-wheels') {
        expect(before[0].labelName).toBe('two-wheels');
        expect(before[0].attributes).toEqual({ [isolated.rideStateAttributeId]: String(rideState) });
        expect(JSON.stringify(before[0].autoFit)).not.toMatch(/bicycle|motorcycle/i);
        if (before[0].autoFit.prototype_key !== null) {
            expect(before[0].autoFit.prototype_key).toMatch(/^two-wheels:[0-9a-f]{12}$/);
        }
    }
    expect(before[0].autoFit).toMatchObject({ request_id: payload.requestId, quality: result.quality,
        class_key: label === 'vehicles' ? `vehicles/${vehicleType}` : label });
    await page.locator('.cvat-annotation-header-undo-button').click();
    await expect(page.locator(objectsSelector)).toHaveCount(0);
    await page.locator('.cvat-annotation-header-redo-button').click();
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    await saveReload(page, isolated);
    expect(await geometry(page)).toEqual(before);
    await writeFile(info.outputPath('lifecycle.json'), JSON.stringify({ caseId: isolated.selected.case_id, mode,
        annotationCount: 1, undoRedo: 'passed', saveReload: 'passed', before, after: await geometry(page) }, null, 2));
    if (result.autoFit.experimental || result.quality === 'estimated') {
        await expect(page.getByTestId('autofit-provenance')).toBeVisible();
    }
    // This smoke verifies one live interaction, not accuracy or the full acceptance gates.
});
