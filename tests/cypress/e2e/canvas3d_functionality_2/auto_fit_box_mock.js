// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

const canvasSelector = '.cvat-canvas3d-perspective canvas';
const objects = '.cvat-objects-sidebar-state-item';

function createIsolatedTask(fixturePath, labels, taskName = `Auto Fit PR1 isolated ${Date.now()}`) {
    return cy.readFile(fixturePath, null).then((buffer) => cy.window().then(async (win) => {
        const file = new win.File([buffer], 'autofit-fixture.zip', { type: 'application/zip' });
        const task = new win.cvat.classes.Task({
            name: taskName,
            labels,
            image_quality: 70,
            use_zip_chunks: true,
        });
        const created = await task.save({ clientFiles: [file] });
        return { taskId: created.id, jobIds: created.jobs.map((job) => job.id) };
    }));
}

context('Auto Fit Box PR1: point picking and annotation lifecycle', () => {
    let jobId;
    let taskId;
    let imageTask;
    let events;
    let observedCanvas;

    function start(type = 'Shape', enabled = true, mode = 'success', delay = 300) {
        cy.window().then((win) => {
            win.localStorage.setItem('cvat.autofit.mock.mode', mode);
            win.localStorage.setItem('cvat.autofit.mock.delay', String(delay));
        });
        cy.interactControlButton('draw-cuboid');
        cy.get('.cvat-auto-fit-box-checkbox input')[enabled ? 'check' : 'uncheck']();
        cy.get('.cvat-draw-cuboid-popover').contains('button', type).click();
    }

    function confirmPoint(repeats = 1) {
        cy.get(canvasSelector).then(([canvas]) => {
            if (observedCanvas !== canvas) {
                observedCanvas = canvas;
                const counts = events;
                canvas.addEventListener('canvas.autofit', (event) => {
                    if (event.detail.status === 'fitting') counts.fitting++;
                });
                canvas.addEventListener('canvas.drawn', (event) => counts.drawn.push(event.detail));
            }
            const cloud = canvas.scene.children.find((child) => child.isPoints);
            const positions = cloud.geometry.getAttribute('position');
            const { camera } = canvas;
            camera.updateMatrixWorld(true);
            const rect = canvas.getBoundingClientRect();
            let coordinates = null;
            for (let index = 0; index < positions.count; index++) {
                const point = camera.position.clone().fromBufferAttribute(positions, index)
                    .applyMatrix4(cloud.matrixWorld).project(camera);
                if (Math.abs(point.x) < 0.75 && Math.abs(point.y) < 0.75 && Math.abs(point.z) < 1) {
                    coordinates = {
                        clientX: rect.left + ((point.x + 1) * rect.width) / 2,
                        clientY: rect.top + ((1 - point.y) * rect.height) / 2,
                    };
                    break;
                }
            }
            expect(coordinates, 'a genuine visible cloud vertex').not.to.eq(null);
            // Deliberately do not send mousemove; double-click must use its own pointer coordinates.
            for (let n = 0; n < repeats; n++) {
                canvas.dispatchEvent(new canvas.ownerDocument.defaultView.MouseEvent('dblclick', coordinates));
            }
        });
    }

    before(() => {
        cy.visit('/auth/login');
        if (Cypress.env('AUTOFIT_REGISTER')) {
            cy.headlessCreateUser({
                username: Cypress.env('user'),
                password: Cypress.env('password'),
                email: Cypress.env('email'),
                firstName: 'AutoFit',
                lastName: 'Test',
            });
        }
        cy.headlessLogin({ nextURL: '/tasks' });
        const fixturePath = Cypress.env('AUTOFIT_FIXTURE') || 'mounted_file_share/test_canvas3d.zip';
        createIsolatedTask(fixturePath, [{ name: 'Car', type: 'cuboid', attributes: [] },
            { name: 'Van', type: 'cuboid', attributes: [] }]).then((created) => {
            [jobId] = created.jobIds;
            taskId = created.taskId;
            cy.writeFile('cypress/results/autofit-task.json', { taskId, jobId, fixturePath });
        });
        createIsolatedTask('mounted_file_share/image_main_task.zip',
            [{ name: 'Car', type: 'any', attributes: [] }]).then((created) => { imageTask = created; });
        if (Cypress.env('canvas3dTaskName')) {
            createIsolatedTask('mounted_file_share/test_canvas3d.zip',
                [{ name: 'points cloud', type: 'any', attributes: [] }], Cypress.env('canvas3dTaskName'));
        }
    });

    beforeEach(() => {
        events = { fitting: 0, drawn: [] };
        observedCanvas = null;
        // Clear only annotations belonging to the isolated task created by this spec.
        cy.window().then(async (win) => {
            const [job] = await win.cvat.jobs.get({ jobID: jobId });
            expect(job.taskId).to.eq(taskId);
            await job.annotations.clear({ reload: true });
            await job.annotations.clear();
            await job.annotations.save();
        });
        cy.intercept({ method: 'GET', pathname: `/api/jobs/${jobId}` }).as('loadJob');
        cy.visit(`/tasks/${taskId}/jobs/${jobId}`);
        cy.wait('@loadJob');
        cy.get('.cvat-spinner').should('not.exist');
        cy.get(canvasSelector).should(([canvas]) => {
            expect(canvas.scene.children.some((child) => child.isPoints)).to.eq(true);
        });
        cy.get(objects).should('not.exist');
        cy.get('.cvat-draw-cuboid-control').should('not.have.class', 'cvat-disabled-canvas-control');
    });

    it('UX01/02: defaults OFF below Label; manual creation remains available', () => {
        cy.interactControlButton('draw-cuboid');
        cy.get('.cvat-auto-fit-box-checkbox input').should('not.be.checked');
        cy.get('.cvat-auto-fit-mock-indicator').should('contain.text', 'Mock Auto Fit');
        cy.get('.cvat-draw-shape-popover-content').then(([popover]) => {
            const label = popover.querySelector('.cvat-label-selector');
            const checkbox = popover.querySelector('.cvat-auto-fit-box-checkbox');
            expect(label || popover.querySelector('.ant-select')).not.to.eq(null);
            expect(checkbox.closest('.ant-row').previousElementSibling.querySelector('.ant-select')).not.to.eq(null);
        });
        cy.get('.cvat-draw-cuboid-shape-button').click();
        cy.get('.cvat-canvas3d-perspective').trigger('mousemove', 480, 160);
        cy.get('.cvat-canvas3d-perspective').dblclick(480, 160);
        cy.get(objects).should('have.length', 1);
        cy.then(() => expect(events.fitting).to.eq(0));
    });

    ['Shape', 'Track'].forEach((type) => {
        it(`UX04/05/12/13: ${type} commits once after repeated double-clicks, from real XYZ`, () => {
            start(type, true, 'success', 700);
            confirmPoint(4);
            cy.get('.cvat-auto-fit-status').should('contain.text', 'in progress');
            cy.get(objects).should('not.exist');
            cy.get(objects).should('have.length', 1);
            cy.get(canvasSelector).should(([canvas]) => {
                expect(canvas.getDrawnObjects()).to.have.length(1);
            }).then(([canvas]) => {
                expect(events.fitting).to.eq(1);
                expect(events.drawn).to.have.length(1);
                const { autoFit } = events.drawn[0];
                const cloud = canvas.scene.children.find((child) => child.isPoints);
                const xyz = cloud.geometry.getAttribute('position');
                expect([autoFit.seed.x, autoFit.seed.y, autoFit.seed.z])
                    .to.deep.eq([xyz.getX(autoFit.seed.pointIndex), xyz.getY(autoFit.seed.pointIndex),
                        xyz.getZ(autoFit.seed.pointIndex)]);
                const [drawn] = canvas.getDrawnObjects();
                expect(drawn.state.objectType).to.eq(type.toLowerCase());
                expect(drawn.state.label.name).to.eq('Car');
                expect(drawn.state.frame).to.eq(0);
                expect(drawn.state.points.slice(6, 9)).to.deep.eq([4, 1.8, 1.6]);
            });
            cy.screenshot(`autofit-${type.toLowerCase()}-success`);
        });
    });

    it('UX03: 2D rectangle and cuboid tools do not show Auto Fit', () => {
        cy.visit(`/tasks/${imageTask.taskId}/jobs/${imageTask.jobIds[0]}`);
        cy.get('.cvat-spinner').should('not.exist');
        cy.interactControlButton('draw-rectangle');
        cy.get('.cvat-auto-fit-box-checkbox').should('not.exist');
        cy.get('body').trigger('mousedown');
        cy.interactControlButton('draw-cuboid');
        cy.get('.cvat-auto-fit-box-checkbox').should('not.exist');
    });

    it('UX06: empty-space hit creates no request or annotation; same draw can retry', () => {
        start();
        cy.get(canvasSelector).then(([canvas]) => {
            const rect = canvas.getBoundingClientRect();
            const positions = canvas.scene.children.find((child) => child.isPoints).geometry.getAttribute('position');
            const candidates = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
                .map(([x, y]) => ({ x: x * rect.width, y: y * rect.height, distance: Infinity }));
            for (let index = 0; index < positions.count; index++) {
                const point = canvas.camera.position.clone()
                    .fromBufferAttribute(positions, index).project(canvas.camera);
                if (Math.abs(point.z) >= 1) continue;
                const x = ((point.x + 1) * rect.width) / 2;
                const y = ((1 - point.y) * rect.height) / 2;
                candidates.forEach((candidate, n) => {
                    candidates[n].distance = Math.min(candidate.distance, Math.hypot(x - candidate.x, y - candidate.y));
                });
            }
            const empty = candidates.find((candidate) => candidate.distance > 12);
            expect(empty, 'empty space inside the canvas').not.to.eq(undefined);
            canvas.dispatchEvent(new canvas.ownerDocument.defaultView.MouseEvent('dblclick', {
                clientX: rect.left + empty.x, clientY: rect.top + empty.y,
            }));
        });
        cy.get('.cvat-auto-fit-status').should('contain.text', 'visible LiDAR point');
        cy.then(() => expect(events.fitting).to.eq(0));
        cy.get(objects).should('not.exist');
        confirmPoint();
        cy.get(objects).should('have.length', 1);
    });

    ['no-fit', 'reject', 'nan', 'infinity', 'negative', 'wrong-frame', 'wrong-request', 'wrong-provider']
        .forEach((mode) => {
            it(`UX07/11: ${mode} rejects with zero annotations and recovers for retry`, () => {
                start('Shape', true, mode);
                confirmPoint();
                cy.get('.cvat-auto-fit-status').should('contain.text', 'retry');
                cy.get(objects).should('not.exist');
                cy.then(() => expect(events.drawn).to.have.length(0));
                if (mode === 'reject') {
                    confirmPoint();
                    cy.get('.cvat-auto-fit-status').should('contain.text', 'retry');
                    cy.then(() => expect(events.fitting).to.eq(2));
                    cy.get(objects).should('not.exist');
                }
                start();
                confirmPoint();
                cy.get(objects).should('have.length', 1);
            });
        });

    it('UX08: Escape cancels a slow result', () => {
        start('Shape', true, 'success', 1200);
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'in progress');
        cy.get('body').type('{esc}');
        cy.get('.cvat-auto-fit-status').should('contain.text', 'cancelled');
        cy.wait(1400); // Wait beyond provider delay to assert no late creation.
        cy.get(objects).should('not.exist');
        cy.then(() => expect(events.drawn).to.have.length(0));
    });

    it('UX09: navigating to another frame drops a slow response', () => {
        start('Shape', true, 'success', 1200);
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'in progress');
        cy.get('.cvat-player-next-button').click();
        cy.wait(1400);
        cy.get(objects).should('not.exist');
        cy.then(() => expect(events.drawn).to.have.length(0));
    });

    it('UX10: unmount drops a slow response', () => {
        start('Shape', true, 'success', 5000);
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'in progress');
        cy.get('.cvat-annotation-header-button').contains('Menu').click();
        cy.contains('Open the task').click();
        cy.url().should('eq', `${Cypress.config('baseUrl')}/tasks/${taskId}`);
        cy.wait(5500);
        cy.then(() => expect(events.drawn).to.have.length(0));
    });

    it('UX10: changing label and object type cancels the previous request', () => {
        start('Shape', true, 'success', 1200);
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'in progress');
        cy.get('.cvat-draw-cuboid-control').click();
        cy.interactControlButton('draw-cuboid');
        cy.switchLabel('Van', 'draw-cuboid');
        cy.get('.cvat-draw-cuboid-track-button').click();
        cy.wait(1400);
        cy.get(objects).should('not.exist');
        cy.then(() => expect(events.drawn).to.have.length(0));
        confirmPoint();
        cy.get(objects).should('have.length', 1);
        cy.get(canvasSelector).then(([canvas]) => {
            const [drawn] = canvas.getDrawnObjects();
            expect(drawn.state.label.name).to.eq('Van');
            expect(drawn.state.objectType).to.eq('track');
        });
    });

    it('provider deadline recovers without creating an annotation', () => {
        start('Shape', true, 'timeout');
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'timed out');
        cy.get(objects).should('not.exist');
        cy.then(() => expect(events.drawn).to.have.length(0));
    });

    it('failed fit preserves an existing cuboid', () => {
        start();
        confirmPoint();
        cy.get(objects).should('have.length', 1);
        let original;
        cy.get(canvasSelector).then(([canvas]) => {
            original = [...canvas.getDrawnObjects()[0].state.points];
        });
        start('Shape', true, 'reject');
        confirmPoint();
        cy.get('.cvat-auto-fit-status').should('contain.text', 'retry');
        cy.get(objects).should('have.length', 1);
        cy.get(canvasSelector).then(([canvas]) => {
            expect(canvas.getDrawnObjects()[0].state.points).to.deep.eq(original);
        });
    });

    ['Shape', 'Track'].forEach((type) => {
        it(`UX14/15: ${type} Undo/Redo and Save/Reload preserve one cuboid and its geometry`, () => {
            start(type);
            confirmPoint();
            cy.get(objects).should('have.length', 1);
            let points;
            cy.get(canvasSelector).should(([canvas]) => {
                expect(events.drawn).to.have.length(1);
                expect(canvas.getDrawnObjects()[0].state.objectType).to.eq(type.toLowerCase());
            }).then(([canvas]) => { points = [...canvas.getDrawnObjects()[0].state.points]; });
            cy.get('.cvat-annotation-header-undo-button').should('have.css', 'pointer-events', 'auto').click();
            cy.get(objects).should('not.exist');
            cy.get('.cvat-annotation-header-redo-button').should('have.css', 'pointer-events', 'auto').click();
            cy.get(objects).should('have.length', 1);
            cy.saveJob();
            cy.reload();
            cy.get(objects).should('have.length', 1);
            cy.get(canvasSelector).should(([canvas]) => {
                expect(canvas.getDrawnObjects()).to.have.length(1);
            }).then(([canvas]) => {
                expect(canvas.getDrawnObjects()[0].state.points).to.deep.eq(points);
            });
            cy.screenshot(`autofit-${type.toLowerCase()}-save-reload`);
        });
    });

    it('UX16: session preference survives repeat drawing; OFF restores manual drawing', () => {
        start();
        confirmPoint();
        cy.get(objects).should('have.length', 1);
        cy.get('body').type('n');
        confirmPoint();
        cy.get(objects).should('have.length', 2);
        start('Shape', false);
        cy.get('.cvat-canvas3d-perspective').dblclick(480, 160);
        cy.get(objects).should('have.length', 3);
        cy.then(() => expect(events.fitting).to.eq(2));
    });
});
