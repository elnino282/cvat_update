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
type VehicleType = 'car' | 'truck' | 'bus' | 'van' | 'other';
const vehicleTypes = new Set<VehicleType>(['car', 'truck', 'bus', 'van', 'other']);
const vehiclePreferences = new WeakMap<object, Map<string, VehicleType>>();
const ridePreferences = new WeakMap<object, Map<string, boolean>>();

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

export function vehicleTypeAttribute(label: any): any | undefined {
    const attribute = label?.attributes?.find((item: any) => item.name === 'type');
    return attribute?.inputType === 'select' && attribute.mutable === false &&
        JSON.stringify(attribute.values) === JSON.stringify(['car', 'truck', 'bus', 'van', 'other']) ? attribute : undefined;
}

export function isVehiclesLabel(job: { labels: any[] }, labelId: number | null): boolean {
    return Boolean(job.labels.some((label) => label.id === labelId &&
        label.name.trim().toLowerCase() === 'vehicles' && vehicleTypeAttribute(label)));
}

export function rideStateAttribute(label: any): any | undefined {
    const attribute = label?.attributes?.find((item: any) => item.name === 'ride_state');
    return attribute?.inputType === 'checkbox' && attribute.mutable === false ? attribute : undefined;
}

export function isTwoWheelsLabel(job: { labels: any[] }, labelId: number | null): boolean {
    return job.labels.some((label) => label.id === labelId &&
        label.name.trim().toLowerCase() === 'two-wheels' && rideStateAttribute(label));
}

export function isAutoFitLabel(job: { labels: any[] }, labelId: number | null): boolean {
    return isCarLabel(job, labelId) || isVehiclesLabel(job, labelId) || isTwoWheelsLabel(job, labelId) ||
        job.labels.some((label) => label.id === labelId && label.name.trim().toLowerCase() === 'pedestrian');
}

export function hasAutoFitLabel(job: { labels: any[] }): boolean {
    return job.labels.some((label) => isAutoFitLabel(job, label.id));
}

const vehiclePreferenceKey = (jobId: number, labelId: number): string => `${jobId}:${labelId}`;

export function getVehicleTypePreference(canvas: object, jobId: number, labelId: number): VehicleType | undefined {
    return vehiclePreferences.get(canvas)?.get(vehiclePreferenceKey(jobId, labelId));
}

export function setVehicleTypePreference(
    canvas: object, jobId: number, labelId: number, value: string | undefined,
): void {
    const current = vehiclePreferences.get(canvas) || new Map<string, VehicleType>();
    const key = vehiclePreferenceKey(jobId, labelId);
    if (value && vehicleTypes.has(value as VehicleType)) current.set(key, value as VehicleType);
    else current.delete(key);
    vehiclePreferences.set(canvas, current);
}

export function vehicleAnnotationAttribute(
    canvas: object, jobId: number, label: any, selectedValue?: VehicleType,
): Record<number, string> | undefined {
    if (label?.name?.trim().toLowerCase() !== 'vehicles') return undefined;
    const spec = vehicleTypeAttribute(label);
    if (!Number.isInteger(spec?.id)) return undefined;
    return {
        [spec.id]: selectedValue || getVehicleTypePreference(canvas, jobId, label.id) ||
            spec.defaultValue || 'other',
    };
}

export function getRideStatePreference(canvas: object, jobId: number, label: any): boolean {
    return ridePreferences.get(canvas)?.get(vehiclePreferenceKey(jobId, label.id)) ??
        (rideStateAttribute(label)?.defaultValue === 'true');
}

export function setRideStatePreference(canvas: object, jobId: number, labelId: number, value: boolean): void {
    const current = ridePreferences.get(canvas) || new Map<string, boolean>();
    current.set(vehiclePreferenceKey(jobId, labelId), value);
    ridePreferences.set(canvas, current);
}

export function rideAnnotationAttribute(
    canvas: object, jobId: number, label: any, selectedValue?: boolean,
): Record<number, string> | undefined {
    if (label?.name?.trim().toLowerCase() !== 'two-wheels') return undefined;
    const spec = rideStateAttribute(label);
    if (!Number.isInteger(spec?.id)) return undefined;
    return { [spec.id]: String(selectedValue ?? getRideStatePreference(canvas, jobId, label)) };
}

export function autoFitPreference(canvas: object): boolean {
    return preferences.get(canvas) || false;
}

export function setAutoFitPreference(canvas: object, enabled: boolean): void {
    preferences.set(canvas, enabled);
}

export function isAutoFitIdentityCurrent(
    request: Pick<FitRequest, 'taskId' | 'jobId' | 'frame' | 'labelId' | 'objectType' | 'vehicleType' |
    'typeAttributeId' | 'rideState' | 'rideStateAttributeId'>,
): boolean {
    const {
        job, player, drawing, canvas,
    } = getCVATStore().getState().annotation;
    if (!job.instance || !isAutoFitLabel(job.instance, request.labelId)) return false;
    const label = job.instance.labels.find((item: any) => item.id === request.labelId);
    let semanticCurrent = request.vehicleType === undefined && request.rideState === undefined &&
        label?.name.trim().toLowerCase() === 'pedestrian';
    if (isCarLabel(job.instance, request.labelId)) semanticCurrent = request.vehicleType === 'car';
    if (isVehiclesLabel(job.instance, request.labelId)) {
        semanticCurrent = request.vehicleType !== 'other' && request.vehicleType !== undefined &&
            request.vehicleType === getVehicleTypePreference(canvas.instance, request.jobId, request.labelId) &&
            vehicleTypeAttribute(label)?.id === request.typeAttributeId;
    }
    if (isTwoWheelsLabel(job.instance, request.labelId)) {
        semanticCurrent = typeof request.rideState === 'boolean' && request.rideState ===
            getRideStatePreference(canvas.instance, request.jobId, label) &&
            rideStateAttribute(label)?.id === request.rideStateAttributeId;
    }
    return semanticCurrent && job.instance.id === request.jobId && job.instance.taskId === request.taskId &&
        player.frame.number === request.frame && !player.frame.fetching &&
        drawing.activeLabelID === request.labelId && drawing.activeObjectType === request.objectType &&
        drawing.activeShapeType === ShapeType.CUBOID && canvas.activeControl === ActiveControl.DRAW_CUBOID;
}

export function autoFitDrawData(
    canvas: unknown, shape: ShapeType, labelId: number, type: ObjectType, vehicleType?: VehicleType,
): AutoFitDraw | undefined {
    const { job: { instance: job }, player: { frame } } = getCVATStore().getState().annotation;
    if (!autoFitAvailable || !(canvas instanceof Canvas3d) || job.dimension !== DimensionType.DIMENSION_3D ||
        !isAutoFitLabel(job, labelId) ||
        shape !== ShapeType.CUBOID || !autoFitPreference(canvas) || !['shape', 'track'].includes(type)) return undefined;
    const selectedLabel = job.labels.find((label: any) => label.id === labelId);
    const isVehicles = selectedLabel?.name.trim().toLowerCase() === 'vehicles';
    let selectedType = vehicleType || getVehicleTypePreference(canvas, job.id, labelId);
    if (!isVehicles) selectedType = isCarLabel(job, labelId) ? 'car' : undefined;
    const typeAttribute = isVehicles ? vehicleTypeAttribute(selectedLabel) : undefined;
    if (isVehicles && (!selectedType || selectedType === 'other' || !typeAttribute || selectedType !==
        getVehicleTypePreference(canvas, job.id, labelId))) return undefined;
    const rideAttribute = isTwoWheelsLabel(job, labelId) ? rideStateAttribute(selectedLabel) : undefined;
    const identity = {
        taskId: job.taskId,
        jobId: job.id,
        frame: frame.number,
        labelId,
        objectType: type as 'shape' | 'track',
        ...(selectedType ? { vehicleType: selectedType } : {}),
        ...(rideAttribute ? {
            rideState: getRideStatePreference(canvas, job.id, selectedLabel),
            rideStateAttributeId: rideAttribute.id,
        } : {}),
        ...(typeAttribute ? { typeAttributeId: typeAttribute.id } : {}),
    };
    let mode: MockMode = 'success';
    let delay = 300;
    if (mockAutoFitEnabled) {
        // Dev fixtures only; production never reads these overrides or creates a mock provider.
        mode = window.localStorage.getItem('cvat.autofit.mock.mode') as MockMode || 'success';
        const configuredDelay = Number(window.localStorage.getItem('cvat.autofit.mock.delay') ?? 300);
        if (Number.isFinite(configuredDelay)) delay = configuredDelay;
    }
    let provider;
    if (mockAutoFitEnabled) {
        provider = createMockAutoFitProvider(mode, delay);
    } else {
        provider = createRealAutoFitProvider(CVAT_AUTOFIT_BACKEND_KIND, canvas);
    }
    return {
        identity,
        allowEstimated: true,
        provider,
        isCurrent: () => {
            const current = getCVATStore().getState().annotation;
            return current.canvas.instance === canvas &&
                isAutoFitIdentityCurrent(identity);
        },
    };
}
