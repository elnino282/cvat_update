// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import {
    AutoFitDraw, Canvas3d, createMockAutoFitProvider, MockMode, FitRequest,
} from 'cvat-canvas3d-wrapper';
import { DimensionType, ObjectType, ShapeType } from 'cvat-core-wrapper';
import { getCVATStore } from 'cvat-store';
import { ActiveControl } from 'reducers';

// Session-only preference. Job metadata updates can replace the Job object during the same session.
// A fresh annotation canvas always starts OFF; never stored in production preferences.
const preferences = new WeakMap<object, boolean>();

// A dedicated compile-time constant avoids dotenv redefining the requested environment flag in production.
declare const CVAT_AUTOFIT_MOCK_BUILD_ENABLED: boolean;
export const mockAutoFitEnabled = process.env.NODE_ENV === 'development' && CVAT_AUTOFIT_MOCK_BUILD_ENABLED;

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
        player.frame.number === request.frame && !player.frame.fetching &&
        drawing.activeLabelID === request.labelId && drawing.activeObjectType === request.objectType &&
        drawing.activeShapeType === ShapeType.CUBOID && canvas.activeControl === ActiveControl.DRAW_CUBOID;
}

export function autoFitDrawData(
    canvas: unknown, shape: ShapeType, labelId: number, type: ObjectType,
): AutoFitDraw | undefined {
    const { job: { instance: job }, player: { frame } } = getCVATStore().getState().annotation;
    if (!(canvas instanceof Canvas3d) || job.dimension !== DimensionType.DIMENSION_3D ||
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
        provider: mockAutoFitEnabled ? createMockAutoFitProvider(mode, delay) : {
            kind: 'snap',
            async fit(request) {
                return {
                    status: 'no-fit', requestId: request.requestId, frame: request.frame, reason: 'unavailable',
                };
            },
        },
        isCurrent: () => {
            const current = getCVATStore().getState().annotation;
            return current.canvas.instance === canvas &&
                isAutoFitIdentityCurrent(identity);
        },
    };
}
