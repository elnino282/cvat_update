import { expect } from 'playwright/test';

export const canvasSelector = '.cvat-canvas3d-perspective canvas';
export const objectsSelector = '.cvat-objects-sidebar-state-item';

export async function openJob(page, fixture) {
    await page.goto(`/tasks/${fixture.id}/jobs/${fixture.jobId}`);
    await expect(page.locator(canvasSelector)).toBeVisible();
    await expect.poll(() => page.locator(canvasSelector).evaluate((canvas) => (
        Boolean(canvas.scene?.children.some((child) => child.isPoints))
    ))).toBe(true);
    await expect(page.locator('.cvat-auto-fit-mock-indicator')).toHaveCount(0);
    await expect(page.locator(objectsSelector)).toHaveCount(0);
}

export async function startDrawing(page, mode = 'Shape', vehicleType = 'car', autoFit = true, options = {}) {
    const { label = 'vehicles', rideState = false } = options;
    await page.keyboard.press('Escape');
    await page.locator('.cvat-draw-cuboid-control').click();
    const popover = page.locator('.cvat-draw-cuboid-popover');
    await expect(popover).toBeVisible();
    await popover.locator('.ant-select').first().click();
    await page.locator('.ant-select-item-option').filter({ hasText: new RegExp(`^${label}$`) }).click();
    const checkbox = page.locator('.cvat-auto-fit-box-checkbox input');
    await expect(checkbox).toBeEnabled();
    await checkbox.setChecked(autoFit);
    if (autoFit && label === 'vehicles') {
        await page.locator('.cvat-auto-fit-vehicle-type').click();
        await page.locator('.ant-select-item-option').filter({ hasText: new RegExp(`^${vehicleType}$`) }).click();
    }
    if (autoFit && label === 'two-wheels') {
        await page.locator('.cvat-auto-fit-ride-state input').setChecked(rideState);
    }
    await popover.getByRole('button', { name: mode, exact: true }).click();
}

export async function confirmPoint(page, seed = null, repeats = 1) {
    const screen = await page.locator(canvasSelector).evaluate((canvas, target) => {
        const cloud = canvas.scene.children.find((child) => child.isPoints);
        const positions = cloud.geometry.getAttribute('position');
        const { camera } = canvas;
        cloud.updateMatrixWorld(true);
        // Use a fixed camera offset relative to fixture XYZ, not machine-specific pixels.
        cloud.geometry.computeBoundingSphere();
        const focus = target ? camera.position.clone().set(...target.point) : cloud.geometry.boundingSphere.center.clone();
        focus.applyMatrix4(cloud.matrixWorld);
        camera.up.set(0, 0, 1);
        if (target) {
            // nuScenes vertices are first returns from this LiDAR origin. Zoom reduces the picker
            // cone without moving through foreground objects; near clipping rejects origin artifacts.
            camera.position.set(0, 0, 0).applyMatrix4(cloud.matrixWorld);
            camera.near = 0.1;
            camera.zoom = 4;
            if (target.cameraView === 'close-up') {
                // An explicit reproducible alternate view of the SAME frozen vertex, for returns
                // whose projected point footprints overlap from the LiDAR origin. No fit feedback.
                camera.position.copy(focus).add(camera.position.clone().set(-1, 1, 1).normalize());
                camera.zoom = 1;
            }
        } else camera.position.copy(focus).add(camera.position.clone().set(12, -12, 10));
        camera.lookAt(focus);
        camera.updateMatrixWorld(true);
        camera.updateProjectionMatrix();
        const rect = canvas.getBoundingClientRect();
        const indices = target ? [target.pointIndex] : Array.from({ length: positions.count }, (_, index) => index);
        for (const index of indices) {
            const point = camera.position.clone().fromBufferAttribute(positions, index)
                .applyMatrix4(cloud.matrixWorld).project(camera);
            if (Math.abs(point.x) < 0.8 && Math.abs(point.y) < 0.8 && Math.abs(point.z) < 1) {
                const screen = { x: rect.left + (point.x + 1) * rect.width / 2,
                    y: rect.top + (1 - point.y) * rect.height / 2, dispatched: Boolean(target) };
                if (target) {
                    // CameraControls restores its camera on the next RAF. Dispatch in the same
                    // projection turn, as the existing real-SNAP Cypress fixture does. This reaches
                    // the normal canvas gesture handler; mocked UX tests use native page.mouse below.
                    canvas.dispatchEvent(new MouseEvent('dblclick', { clientX: screen.x, clientY: screen.y,
                        button: 0, detail: 2, bubbles: true }));
                }
                return screen;
            }
        }
        throw new Error('Fixture has no visible LiDAR vertex in normalized camera view');
    }, seed);
    if (!screen.dispatched) {
        for (let index = 0; index < repeats; index++) await page.mouse.dblclick(screen.x, screen.y);
    }
}

export async function geometry(page) {
    return page.locator(canvasSelector).evaluate((canvas) => canvas.getDrawnObjects().map(({ state }) => ({
        points: [...state.points], objectType: state.objectType, labelName: state.label.name,
        attributes: { ...state.attributes }, autoFit: { ...state.autoFit },
    })));
}

export async function interceptAutoFit(page, fixture, handler) {
    const requests = [];
    await page.route(`**/api/jobs/${fixture.jobId}/auto-fit**`, async (route) => {
        const { pathname } = new URL(route.request().url());
        if (pathname.endsWith('/prepare')) {
            const data = route.request().method() === 'POST' ? route.request().postDataJSON() :
                Object.fromEntries(new URL(route.request().url()).searchParams);
            return route.fulfill({ json: { frameToken: 'a'.repeat(32), frame: Number(data.frame), state: 'ready' } });
        }
        if (route.request().method() === 'DELETE') return route.fulfill({ status: 204 });
        const payload = route.request().postDataJSON();
        requests.push(payload);
        return handler(route, payload);
    });
    return requests;
}

export function successfulResponse(payload, quality = 'confident') {
    return { status: 'ok', quality, requestId: payload.requestId, frame: payload.frame,
        provider: process.env.AUTOFIT_BACKEND || 'snap', center: [payload.seed.x, payload.seed.y, payload.seed.z],
        size: [4, 1.8, 1.6], rotation: [0, 0, 0], confidence: quality === 'confident' ? 0.95 : 0.4,
        autoFit: { version: 1, request_id: payload.requestId, frame: payload.frame,
            class_key: payload.vehicleType ? `vehicles/${payload.vehicleType}` :
                typeof payload.rideState === 'boolean' ? 'two-wheels' : 'pedestrian',
            quality, fit_mode: quality === 'estimated' ? 'hybrid_prior' : 'geometry',
            yaw_uncertain: quality === 'estimated', prototype_key: quality === 'estimated' ? 'fixture-only' : null,
            prior_version: quality === 'estimated' ? 'mocked-ui-only' : null, experimental: payload.vehicleType !== 'car' } };
}


export async function saveReload(page, fixture, count = 1) {
    const saved = page.waitForResponse((response) => response.url().includes(`/api/jobs/${fixture.jobId}/annotations`) &&
        ['PATCH', 'PUT'].includes(response.request().method()) && response.ok());
    await page.locator('.cvat-annotation-header-save-button').click();
    await saved;
    await page.reload();
    await expect(page.locator(canvasSelector)).toBeVisible();
    await expect(page.locator(objectsSelector)).toHaveCount(count);
}


export async function pickedSeedEvidence(page, seed, selected) {
    const vertex = await page.locator(canvasSelector).evaluate((canvas, pointIndex) => {
        const cloud = canvas.scene.children.find((child) => child.isPoints);
        const positions = cloud.geometry.getAttribute('position');
        if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= positions.count) return null;
        return [positions.getX(pointIndex), positions.getY(pointIndex), positions.getZ(pointIndex)];
    }, seed.pointIndex);
    expect(vertex, 'Payload index must identify a rendered finite source vertex').not.toBeNull();
    expect(vertex.every(Number.isFinite)).toBe(true);
    const actual = [seed.x, seed.y, seed.z];
    for (let axis = 0; axis < 3; axis++) expect(Math.abs(vertex[axis] - actual[axis])).toBeLessThanOrEqual(1e-5);
    const axes = selected.gt_rotation_matrix;
    expect(axes).toHaveLength(3);
    const centered = actual.map((value, axis) => value - selected.gt_center[axis]);
    const local = [0, 1, 2].map((axis) => centered.reduce((sum, value, row) => sum + value * axes[row][axis], 0));
    for (let axis = 0; axis < 3; axis++) {
        expect(Math.abs(local[axis]), `Picked LiDAR vertex must remain in intended instance on axis ${axis}`)
            .toBeLessThanOrEqual(selected.gt_size[axis] / 2 + 0.05);
    }
    return { pointIndex: seed.pointIndex, vertex, local, toleranceMetres: 0.05 };
}
