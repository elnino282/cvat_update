// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

const productionGate = Cypress.env('AUTOFIT_GATE_TASK_ID') && Cypress.env('AUTOFIT_GATE_JOB_ID');

(productionGate ? context : context.skip)('Auto Fit Box production gate: no mock fallback', () => {
    it('ignores development fixture overrides and creates zero cuboids when real AI is unavailable', () => {
        const taskId = Number(Cypress.env('AUTOFIT_GATE_TASK_ID'));
        const jobId = Number(Cypress.env('AUTOFIT_GATE_JOB_ID'));
        expect(taskId, 'isolated task from the preceding mock run').to.be.greaterThan(0);
        expect(jobId, 'isolated job from the preceding mock run').to.be.greaterThan(0);
        cy.prepareUserSession();
        cy.window().then(async (win) => {
            const [job] = await win.cvat.jobs.get({ jobID: jobId });
            expect(job.taskId).to.eq(taskId);
            const [task] = await win.cvat.tasks.get({ id: taskId });
            expect(task.name).to.match(/^Auto Fit PR1 isolated /);
            const annotations = await job.annotations.get(0);
            expect(annotations).to.have.length(0);
        });
        cy.intercept({ method: 'GET', pathname: `/api/jobs/${jobId}` }).as('loadJob');
        cy.visit(`/tasks/${taskId}/jobs/${jobId}`);
        cy.wait('@loadJob');
        cy.get('.cvat-spinner').should('not.exist');
        cy.get('.cvat-canvas3d-perspective canvas').should(([canvas]) => {
            expect(canvas.scene.children.some((child) => child.isPoints)).to.eq(true);
        });
        cy.get('.cvat-objects-sidebar-state-item').should('not.exist');
        cy.window().then((win) => {
            win.localStorage.setItem('cvat.autofit.mock.mode', 'success');
            win.localStorage.setItem('cvat.autofit.mock.delay', '0');
        });
        cy.interactControlButton('draw-cuboid');
        cy.get('.cvat-auto-fit-mock-indicator').should('not.exist');
        cy.get('.cvat-auto-fit-box-checkbox input').should('not.be.checked').check();
        cy.get('.cvat-draw-cuboid-shape-button').click();
        cy.get('.cvat-canvas3d-perspective canvas').then(([canvas]) => {
            let commits = 0;
            canvas.addEventListener('canvas.drawn', () => { commits++; });
            const cloud = canvas.scene.children.find((child) => child.isPoints);
            const positions = cloud.geometry.getAttribute('position');
            const { camera } = canvas;
            camera.updateMatrixWorld(true);
            const rect = canvas.getBoundingClientRect();
            let coordinates;
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
            expect(coordinates, 'genuine LiDAR seed').not.to.eq(undefined);
            canvas.dispatchEvent(new canvas.ownerDocument.defaultView.MouseEvent('dblclick', coordinates));
            cy.get('.cvat-auto-fit-status').should('contain.text', 'Auto Fit is unavailable').then(() => {
                expect(commits).to.eq(0);
                expect(canvas.getDrawnObjects()).to.have.length(0);
            });
        });
        cy.get('.cvat-objects-sidebar-state-item').should('not.exist');
        cy.get('.cvat-annotation-header-undo-button').should('have.css', 'pointer-events', 'none');
        cy.screenshot('autofit-production-unavailable-no-mock');
    });
});
