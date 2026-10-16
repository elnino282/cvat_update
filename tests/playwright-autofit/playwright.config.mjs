import { defineConfig, devices } from 'playwright/test';

export default defineConfig({
    testDir: './specs',
    timeout: 60_000,
    expect: { timeout: 15_000 },
    fullyParallel: false,
    workers: 1, // The real service/GPU and each project task are shared resources.
    retries: 0,
    forbidOnly: Boolean(process.env.CI),
    reporter: [['list'], ['html', { open: 'never' }], ['json', { outputFile: 'test-results/results.json' }]],
    use: {
        ...devices['Desktop Chrome'],
        ...(process.env.AUTOFIT_BROWSER_CHANNEL ? { channel: process.env.AUTOFIT_BROWSER_CHANNEL } : {}),
        baseURL: process.env.CVAT_BASE_URL || 'http://localhost:8080',
        screenshot: 'only-on-failure',
        trace: 'retain-on-failure',
        video: process.env.AUTOFIT_VIDEO === 'off' ? 'off' : 'retain-on-failure',
        launchOptions: { args: ['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader'] },
    },
    projects: [
        { name: 'autofit-ui-mocked', testMatch: 'ui-mocked.spec.mjs' },
        { name: 'autofit-real-nuscenes', testMatch: 'real-nuscenes.spec.mjs' },
    ],
});
