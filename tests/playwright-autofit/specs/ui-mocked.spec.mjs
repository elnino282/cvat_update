import { test, expect } from '../support/fixtures.mjs';
import { openJob, startDrawing, confirmPoint, geometry, interceptAutoFit, successfulResponse,
    objectsSelector, saveReload } from '../support/ui.mjs';

for (const mode of ['Shape', 'Track']) {
    test(`PW-03/04/10/11/12 ${mode}: one request, undo/redo, persisted geometry and type`, async ({ page, isolated }) => {
        const requests = await interceptAutoFit(page, isolated, (route, payload) => (
            route.fulfill({ json: successfulResponse(payload) })
        ));
        await openJob(page, isolated);
        await startDrawing(page, mode);
        await confirmPoint(page);
        await expect(page.locator(objectsSelector)).toHaveCount(1);
        expect(requests).toHaveLength(1);
        expect(requests[0]).toMatchObject({ taskId: isolated.id, jobId: isolated.jobId,
            labelId: isolated.vehicleLabelId, typeAttributeId: isolated.typeAttributeId,
            vehicleType: 'car', objectType: mode.toLowerCase(), allowEstimated: true });
        expect(requests[0].requestId).toMatch(/^[0-9a-f-]{36}$/);
        const before = await geometry(page);
        expect(before[0].attributes[String(isolated.typeAttributeId)]).toBe('car');
        await page.locator('.cvat-annotation-header-undo-button').click();
        await expect(page.locator(objectsSelector)).toHaveCount(0);
        await page.locator('.cvat-annotation-header-redo-button').click();
        await expect(page.locator(objectsSelector)).toHaveCount(1);
        const saved = page.waitForResponse((response) => response.url().includes(`/api/jobs/${isolated.jobId}/annotations`) &&
            ['PATCH', 'PUT'].includes(response.request().method()) && response.ok());
        await page.locator('.cvat-annotation-header-save-button').click();
        await saved;
        await page.reload();
        await expect(page.locator(objectsSelector)).toHaveCount(1);
        expect(await geometry(page)).toEqual(before);
    });
}

test('PW-05 Estimated metadata and feedback survive Save/Reload', async ({ page, isolated }) => {
    await interceptAutoFit(page, isolated, (route, payload) => route.fulfill({ json: successfulResponse(payload, 'estimated') }));
    await openJob(page, isolated);
    await startDrawing(page);
    await confirmPoint(page);
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    await expect(page.getByTestId('autofit-provenance')).toContainText('Estimated');
    const before = await geometry(page);
    expect(before[0].autoFit.quality).toBe('estimated');
    await saveReload(page, isolated);
    expect(await geometry(page)).toEqual(before);
    await expect(page.getByTestId('autofit-provenance')).toContainText('Estimated');
});

for (const vehicleType of ['car', 'truck', 'bus', 'van']) {
    test(`PW-02 runtime attribute mapping: ${vehicleType}`, async ({ page, isolated }) => {
        const requests = await interceptAutoFit(page, isolated, (route, payload) => route.fulfill({ json: {
            status: 'no-fit', requestId: payload.requestId, frame: payload.frame,
            provider: process.env.AUTOFIT_BACKEND || 'snap', reason: 'unsupported-type',
        } }));
        await openJob(page, isolated);
        await startDrawing(page, 'Shape', vehicleType);
        await confirmPoint(page);
        await expect.poll(() => requests.length).toBe(1);
        expect(requests[0]).toMatchObject({ labelId: isolated.vehicleLabelId,
            typeAttributeId: isolated.typeAttributeId, vehicleType });
        await expect(page.locator(objectsSelector)).toHaveCount(0);
    });
}

test('PW-08/09 frame change drops a held response; repeated gesture makes one request', async ({ page, isolated }) => {
    let resolveHeld;
    let resolveHandled;
    const held = new Promise((resolve) => { resolveHeld = resolve; });
    const handled = new Promise((resolve) => { resolveHandled = resolve; });
    const requests = await interceptAutoFit(page, isolated, async (route, payload) => {
        await held;
        await route.fulfill({ json: successfulResponse(payload) }).catch(() => {}); // Aborted request is expected.
        resolveHandled();
    });
    await openJob(page, isolated);
    await startDrawing(page);
    await confirmPoint(page, null, 3);
    await expect.poll(() => requests.length).toBe(1);
    await expect(page.locator('.cvat-auto-fit-status')).toContainText('in progress');
    await page.locator('.cvat-player-next-button').click();
    await expect(page.locator('.cvat-player-frame-selector input')).toHaveValue('1');
    resolveHeld();
    await handled;
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await expect(page.locator('.cvat-auto-fit-status')).toHaveCount(0);
    await expect(page.locator(objectsSelector)).toHaveCount(0);
});

test('PW-07 timeout response recovers the control without annotation', async ({ page, isolated }) => {
    await interceptAutoFit(page, isolated, (route, payload) => route.fulfill({ json: {
        status: 'no-fit', requestId: payload.requestId, frame: payload.frame,
        provider: process.env.AUTOFIT_BACKEND || 'snap', reason: 'timeout',
    } }));
    await openJob(page, isolated);
    await startDrawing(page);
    await confirmPoint(page);
    await expect(page.locator('.cvat-auto-fit-status')).toContainText('timed out');
    await expect(page.locator(objectsSelector)).toHaveCount(0);
});


for (const [label, rideState, mode] of [['pedestrian', false, 'Shape'], ['two-wheels', true, 'Track']]) {
    test(`PW-05/10/11 ${label} ${mode}: runtime schema and estimated provenance persist`, async ({ page, isolated }) => {
        const requests = await interceptAutoFit(page, isolated, (route, payload) => (
            route.fulfill({ json: successfulResponse(payload, 'estimated') })
        ));
        await openJob(page, isolated);
        await startDrawing(page, mode, undefined, true, { label, rideState });
        await confirmPoint(page);
        await expect(page.locator(objectsSelector)).toHaveCount(1);
        expect(requests).toHaveLength(1);
        expect(requests[0].vehicleType).toBeUndefined();
        expect(requests[0]).toMatchObject({ coordinateSystem: 'lidar', units: 'metres',
            labelId: label === 'pedestrian' ? isolated.pedestrianLabelId : isolated.twoWheelsLabelId });
        if (label === 'two-wheels') expect(requests[0]).toMatchObject({ rideState: true,
            rideStateAttributeId: isolated.rideStateAttributeId });
        const before = await geometry(page);
        expect(before[0].autoFit).toMatchObject({ class_key: label, experimental: true, quality: 'estimated' });
        if (label === 'two-wheels') expect(before[0].attributes[String(isolated.rideStateAttributeId)]).toBe('true');
        await expect(page.getByTestId('autofit-provenance')).toContainText('Estimated · Experimental');
        await page.locator('.cvat-annotation-header-undo-button').click();
        await expect(page.locator(objectsSelector)).toHaveCount(0);
        await page.locator('.cvat-annotation-header-redo-button').click();
        await expect(page.locator(objectsSelector)).toHaveCount(1);
        await saveReload(page, isolated);
        expect(await geometry(page)).toEqual(before);
        await expect(page.getByTestId('autofit-provenance')).toContainText('Estimated · Experimental');
    });
}

test('PW-18 Other stays manual without an inference request', async ({ page, isolated }) => {
    const requests = await interceptAutoFit(page, isolated, () => { throw new Error('Manual Other called Auto Fit'); });
    await openJob(page, isolated);
    await startDrawing(page, 'Shape', 'other');
    await confirmPoint(page);
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    expect(requests).toHaveLength(0);
    const [state] = await geometry(page);
    expect(state.attributes[String(isolated.typeAttributeId)]).toBe('other');
    expect(state.autoFit).toEqual({});
});

test('PW-07 a provider response after the deadline cannot create a box', async ({ page, isolated }) => {
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    const requests = await interceptAutoFit(page, isolated, async (route, payload) => {
        await held;
        await route.fulfill({ json: successfulResponse(payload) }).catch(() => {});
    });
    await openJob(page, isolated);
    await startDrawing(page);
    await confirmPoint(page);
    await expect.poll(() => requests.length).toBe(1);
    await expect(page.locator('.cvat-auto-fit-status')).toContainText('timed out');
    release();
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await expect(page.locator(objectsSelector)).toHaveCount(0);
});

test('PW-13 copied Vehicles cuboid retains source type and creation provenance', async ({ page, isolated }) => {
    const requests = await interceptAutoFit(page, isolated, (route, payload) => (
        route.fulfill({ json: successfulResponse(payload, 'estimated') })
    ));
    await openJob(page, isolated);
    await startDrawing(page, 'Shape', 'truck');
    await confirmPoint(page);
    await expect(page.locator(objectsSelector)).toHaveCount(1);
    const [original] = await geometry(page);
    // Change only the next-drawing preference, then copy the existing object.
    await startDrawing(page, 'Shape', 'bus');
    await page.keyboard.press('Escape');
    await page.locator(objectsSelector).first().locator('[aria-label="more"]').click();
    await page.locator('.cvat-object-item-menu-make-copy').click();
    await confirmPoint(page);
    await expect(page.locator(objectsSelector)).toHaveCount(2);
    expect(requests).toHaveLength(1);
    const states = await geometry(page);
    for (const state of states) {
        expect(state.attributes[String(isolated.typeAttributeId)]).toBe('truck');
        expect(state.autoFit).toEqual(original.autoFit);
    }
    await saveReload(page, isolated, 2);
    expect(await geometry(page)).toEqual(states);
});
