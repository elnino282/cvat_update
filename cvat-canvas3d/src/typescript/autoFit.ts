// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

export type Triple = [number, number, number];

export interface FitIdentity {
    taskId: number;
    jobId: number;
    frame: number;
    labelId: number;
    objectType: 'shape' | 'track';
    vehicleType?: 'car' | 'truck' | 'bus' | 'van' | 'other';
    typeAttributeId?: number;
    rideState?: boolean;
    rideStateAttributeId?: number;
}

export interface FitSeed {
    x: number;
    y: number;
    z: number;
    // Index in the finite XYZ geometry rendered for this frame, after PCD NaN filtering.
    pointIndex: number;
}

export interface FitRequest extends FitIdentity {
    requestId: string;
    requestedAt: number;
    seed: FitSeed;
    frameToken?: string;
    allowEstimated?: boolean;
    coordinateSystem?: 'lidar';
    units?: 'metres';
}

export interface FitMetadata {
    version: 1;
    request_id: string;
    frame: number;
    class_key: string;
    quality: 'confident' | 'estimated';
    fit_mode: 'geometry' | 'hybrid_prior';
    yaw_uncertain: boolean;
    prototype_key: string | null;
    prior_version: string | null;
    experimental: boolean;
}

export type FitResult = {
    status: 'ok';
    requestId: string;
    frame: number;
    center: Triple;
    size: Triple;
    rotation: Triple;
    provider: 'mock' | 'snap' | 'geometry-only';
    confidence?: number;
    quality?: 'confident' | 'estimated';
    diagnostics?: Record<string, unknown>;
    autoFit?: FitMetadata;
} | {
    status: 'no-fit';
    requestId: string;
    frame: number;
    reason: 'no-points' | 'ambiguous' | 'insufficient-evidence' | 'unavailable' |
    'timeout' | 'cancelled' | 'invalid-geometry' | 'invalid-seed' | 'frame-expired' | 'unsupported-type' |
    'invalid-coordinates' | 'no-seed-support' | 'unsafe-fit';
    provider?: 'mock' | 'snap' | 'geometry-only';
    diagnostics?: Record<string, unknown>;
};

export interface AutoFitProvider {
    readonly kind: 'mock' | 'snap' | 'geometry-only';
    fit(request: FitRequest, signal: AbortSignal): Promise<FitResult>;
}

export interface AutoFitDraw {
    identity: FitIdentity;
    provider: AutoFitProvider;
    isCurrent(): boolean;
    allowEstimated?: boolean;
}

function finiteTriple(value: unknown): value is Triple {
    return Array.isArray(value) && value.length === 3 && value.every((n) => typeof n === 'number' && Number.isFinite(n));
}

// CVAT uses frame XYZ in metres, Euler XYZ angles in radians, then scale along x/y/z.
// The trailing seven values preserve the existing 16-value ObjectState representation.
export function fitResultToPoints(result: FitResult, request: FitRequest, provider: AutoFitProvider['kind']): number[] {
    if (!result || result.requestId !== request.requestId || result.frame !== request.frame) {
        throw new Error('Auto Fit response does not match this request');
    }
    const fallback = request.allowEstimated && provider === 'snap' &&
        result.status === 'ok' && result.provider === 'geometry-only' && result.quality === 'estimated';
    if (result.status !== 'ok' || (result.provider !== provider && !fallback) ||
        (result.quality !== undefined && !['confident', 'estimated'].includes(result.quality)) ||
        (result.quality === 'estimated' && !request.allowEstimated) ||
        !['mock', 'snap', 'geometry-only'].includes(result.provider) ||
        !finiteTriple(result.center) || !finiteTriple(result.rotation) || !finiteTriple(result.size) ||
        result.size.some((n) => n <= 0) ||
        (result.confidence !== undefined && (!Number.isFinite(result.confidence) ||
            result.confidence < 0 || result.confidence > 1))) {
        throw new Error('Auto Fit returned invalid cuboid geometry');
    }
    if (result.autoFit) {
        const metadata = result.autoFit;
        if (metadata.version !== 1 || metadata.request_id !== request.requestId || metadata.frame !== request.frame ||
            !['vehicles/car', 'vehicles/truck', 'vehicles/bus', 'vehicles/van', 'pedestrian', 'two-wheels']
                .includes(metadata.class_key) || metadata.quality !== (result.quality || 'confident') ||
            !['geometry', 'hybrid_prior'].includes(metadata.fit_mode) ||
            typeof metadata.yaw_uncertain !== 'boolean' || typeof metadata.experimental !== 'boolean' ||
            !(metadata.prototype_key === null || typeof metadata.prototype_key === 'string') ||
            !(metadata.prior_version === null || typeof metadata.prior_version === 'string') ||
            (request.vehicleType !== undefined && metadata.class_key !== `vehicles/${request.vehicleType}`) ||
            (request.rideState !== undefined && metadata.class_key !== 'two-wheels')) {
            throw new Error('Auto Fit returned invalid provenance');
        }
    }
    return [...result.center, ...result.rotation, ...result.size, 0, 0, 0, 0, 0, 0, 0];
}

export type FitOutcome = { status: 'ok'; request: FitRequest; points: number[];
    quality: 'confident' | 'estimated'; autoFit?: FitMetadata } |
{ status: 'retry'; hint: string } | { status: 'cancelled' };

function requestUUID(): string {
    // randomUUID is restricted to secure contexts. getRandomValues also works on
    // explicitly configured LAN development origins.
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] % 16) + 64;
    bytes[8] = (bytes[8] % 64) + 128;
    const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

// Owns the synchronous gesture guard, cancellation and provider deadline. Independent of React rendering.
export class AutoFitOperation {
    private pending: { abort: AbortController; timer: ReturnType<typeof setTimeout> } | null = null;

    public get inFlight(): boolean {
        return this.pending !== null;
    }

    public cancel(): void {
        const { pending } = this;
        this.pending = null;
        if (pending) {
            clearTimeout(pending.timer);
            pending.abort.abort();
        }
    }

    public async start(draw: AutoFitDraw, seed: FitSeed, timeout = 2000): Promise<FitOutcome> {
        if (this.pending || !draw.isCurrent()) return { status: 'cancelled' };
        if (![seed.x, seed.y, seed.z].every(Number.isFinite) ||
            !Number.isInteger(seed.pointIndex) || seed.pointIndex < 0) {
            return { status: 'retry', hint: 'Double-click a visible LiDAR point to retry.' };
        }
        const request: FitRequest = {
            ...draw.identity,
            seed: { ...seed },
            requestId: requestUUID(),
            requestedAt: Date.now(),
            allowEstimated: draw.allowEstimated || false,
            coordinateSystem: 'lidar',
            units: 'metres',
        };
        const abort = new AbortController();
        let timedOut = false;
        const pending = {
            abort,
            timer: setTimeout(() => {
                timedOut = true;
                abort.abort();
            }, timeout),
        };
        this.pending = pending;
        try {
            // Race even providers that ignore AbortSignal, while retaining a rejection handler for late failures.
            const cancelled = new Promise<never>((_, reject) => {
                abort.signal.addEventListener('abort', () => reject(new Error('Cancelled')), { once: true });
            });
            const result = await Promise.race([draw.provider.fit(request, abort.signal), cancelled]);
            if (this.pending !== pending || !draw.isCurrent()) return { status: 'cancelled' };
            if (result?.requestId !== request.requestId || result?.frame !== request.frame) {
                throw new Error('Mismatched response');
            }
            if (result.status === 'no-fit') {
                let hint = 'No safe fit. Double-click another LiDAR point to retry.';
                if (result.reason === 'timeout') hint = 'Auto Fit timed out. Double-click a LiDAR point to retry.';
                if (result.reason === 'unavailable') hint = 'Auto Fit is unavailable. Try again later.';
                if (result.reason === 'unsupported-type') hint = 'Auto Fit is not enabled for this class. Use manual drawing.';
                if (result.reason === 'invalid-coordinates') hint = 'Invalid LiDAR coordinates. Reload the frame to retry.';
                if (result.reason === 'no-seed-support') hint = 'No LiDAR support at this seed. Double-click a visible point.';
                if (result.reason === 'unsafe-fit') {
                    hint = result.diagnostics?.reason === 'prior-unavailable' || result.diagnostics?.prior_unavailable ||
                        (Array.isArray(result.diagnostics?.diagnosticFlags) &&
                            result.diagnostics.diagnosticFlags.includes('prior-unavailable')) ?
                        'No safe fit: reliable size prior is unavailable. Use manual drawing.' :
                        'No safe fit for this instance. Try another point or use manual drawing.';
                }
                return {
                    status: 'retry',
                    hint,
                };
            }
            return {
                status: 'ok',
                request,
                points: fitResultToPoints(result, request, draw.provider.kind),
                quality: result.quality || 'confident',
                ...(result.autoFit ? { autoFit: { ...result.autoFit } } : {}),
            };
        } catch (_error) {
            if (this.pending !== pending || !draw.isCurrent()) return { status: 'cancelled' };
            return {
                status: 'retry',
                hint: timedOut ? 'Auto Fit timed out. Double-click a LiDAR point to retry.' :
                    'Auto Fit failed. Double-click a LiDAR point to retry.',
            };
        } finally {
            clearTimeout(pending.timer);
            if (this.pending === pending) this.pending = null;
        }
    }
}
