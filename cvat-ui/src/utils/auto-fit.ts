// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import {
    AutoFitDraw, Canvas3d, createMockAutoFitProvider, MockMode, FitRequest,
} from 'cvat-canvas3d-wrapper';
import { DimensionType, ObjectType, ShapeType } from 'cvat-core-wrapper';
import { getCVATStore } from 'cvat-store';
import { ActiveControl } from 'reducers';
import { createRealAutoFitProvider } from './auto-fit-provider';

// Session-only preference. Job metadata updates can replace the Job object during the same session.
// A fresh annotation canvas always starts OFF; never stored in production preferences.
const preferences = new WeakMap<object, boolean>();

// A dedicated compile-time constant avoids dotenv redefining the requested environment flag in production.
declare const CVAT_AUTOFIT_MOCK_BUILD_ENABLED: boolean;
declare const CVAT_AUTOFIT_REAL_BUILD_ENABLED: boolean;
declare const CVAT_AUTOFIT_BACKEND_KIND: 'snap' | 'geometry-only';
export const mockAutoFitEnabled = process.env.NODE_ENV === 'development' && CVAT_AUTOFIT_MOCK_BUILD_ENABLED;
export const realAutoFitEnabled = CVAT_AUTOFIT_REAL_BUILD_ENABLED &&
    ['snap', 'geometry-only'].includes(CVAT_AUTOFIT_BACKEND_KIND);
export const autoFitAvailable = mockAutoFitEnabled || realAutoFitEnabled;

export function isCarLabel(job: { labels: { id?: number; name: string }[] }, labelId: number | null): boolean {
    return job.labels.some((label) => label.id === labelId && label.name.trim().toLowerCase() === 'car');
}

export function autoFitPreference(canvas: object): boolean {
    return preferences.get(canvas) || false;
}

export function setAutoFitPreference(canvas: object, enabled: boolean): void {
    preferences.set(canvas, enabled);
}

export function isAutoFitIdentityCurrent(
    request: Pick<FitRequest, 'taskId' | 'jobId' | 'frame' | 'labelId' | 'objectType'>,
): boolean {
    const {
        job, player, drawing, canvas,
    } = getCVATStore().getState().annotation;
    return job.instance?.id === request.jobId && job.instance?.taskId === request.taskId &&
        isCarLabel(job.instance, request.labelId) &&
        player.frame.number === request.frame && !player.frame.fetching &&
        drawing.activeLabelID === request.labelId && drawing.activeObjectType === request.objectType &&
        drawing.activeShapeType === ShapeType.CUBOID && canvas.activeControl === ActiveControl.DRAW_CUBOID;
}

export function autoFitDrawData(
    canvas: unknown, shape: ShapeType, labelId: number, type: ObjectType,
): AutoFitDraw | undefined {
    const { job: { instance: job }, player: { frame } } = getCVATStore().getState().annotation;
    if (!autoFitAvailable || !(canvas instanceof Canvas3d) || job.dimension !== DimensionType.DIMENSION_3D ||
        !isCarLabel(job, labelId) ||
        shape !== ShapeType.CUBOID || !autoFitPreference(canvas) || !['shape', 'track'].includes(type)) return undefined;
    const identity = {
        taskId: job.taskId,
        jobId: job.id,
        frame: frame.number,
        labelId,
        objectType: type as 'shape' | 'track',
    };
    let mode: MockMode = 'success';
    let delay = 300;
    if (mockAutoFitEnabled) {
        // Dev fixtures only; production never reads these overrides or creates a mock provider.
        mode = window.localStorage.getItem('cvat.autofit.mock.mode') as MockMode || 'success';
        const configuredDelay = Number(window.localStorage.getItem('cvat.autofit.mock.delay') ?? 300);
        if (Number.isFinite(configuredDelay)) delay = configuredDelay;
    }
    return {
        identity,
        allowEstimated: true,
        provider: mockAutoFitEnabled ? createMockAutoFitProvider(mode, delay) :
            createRealAutoFitProvider(CVAT_AUTOFIT_BACKEND_KIND, canvas),
        isCurrent: () => {
            const current = getCVATStore().getState().annotation;
            return current.canvas.instance === canvas &&
                isAutoFitIdentityCurrent(identity);
        },
    };
}
