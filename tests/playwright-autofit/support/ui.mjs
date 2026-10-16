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

export async function startDrawing(page, mode = 'Shape', vehicleType = 'car', autoFit = true) {
    await page.keyboard.press('Escape');
    await page.locator('.cvat-draw-cuboid-control').click();
    const popover = page.locator('.cvat-draw-cuboid-popover');
    await expect(popover).toBeVisible();
    await popover.locator('.ant-select').first().click();
    await page.locator('.ant-select-item-option').filter({ hasText: /^vehicles$/ }).click();
    const checkbox = page.locator('.cvat-auto-fit-box-checkbox input');
    await expect(checkbox).toBeEnabled();
    await checkbox.setChecked(autoFit);
    if (autoFit) {
        await page.locator('.cvat-auto-fit-vehicle-type').click();
        await page.locator('.ant-select-item-option').filter({ hasText: new RegExp(`^${vehicleType}$`) }).click();
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
        camera.position.copy(focus).add(camera.position.clone().set(12, -12, 10));
        camera.lookAt(focus);
        camera.updateMatrixWorld(true);
        camera.updateProjectionMatrix();
        const rect = canvas.getBoundingClientRect();
        const indices = target ? [target.pointIndex] : Array.from({ length: positions.count }, (_, index) => index);
        for (const index of indices) {
            const point = camera.position.clone().fromBufferAttribute(positions, index)
                .applyMatrix4(cloud.matrixWorld).project(camera);
            if (Math.abs(point.x) < 0.8 && Math.abs(point.y) < 0.8 && Math.abs(point.z) < 1) {
                return { x: rect.left + (point.x + 1) * rect.width / 2,
                    y: rect.top + (1 - point.y) * rect.height / 2 };
            }
        }
        throw new Error('Fixture has no visible LiDAR vertex in normalized camera view');
    }, seed);
    for (let index = 0; index < repeats; index++) await page.mouse.dblclick(screen.x, screen.y);
}

export async function geometry(page) {
    return page.locator(canvasSelector).evaluate((canvas) => canvas.getDrawnObjects().map(({ state }) => ({
        points: [...state.points], objectType: state.objectType, labelName: state.label.name,
        attributes: { ...state.attributes },
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
        size: [4, 1.8, 1.6], rotation: [0, 0, 0], confidence: quality === 'confident' ? 0.95 : 0.4 };
}
