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
} | {
    status: 'no-fit';
    requestId: string;
    frame: number;
    reason: 'no-points' | 'ambiguous' | 'insufficient-evidence' | 'unavailable' |
    'timeout' | 'cancelled' | 'invalid-geometry' | 'invalid-seed' | 'frame-expired' | 'unsupported-type';
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
    return [...result.center, ...result.rotation, ...result.size, 0, 0, 0, 0, 0, 0, 0];
}

export type FitOutcome = { status: 'ok'; request: FitRequest; points: number[]; quality: 'confident' | 'estimated' } |
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
                let hint = 'No confident fit. Double-click another LiDAR point to retry.';
                if (result.reason === 'timeout') hint = 'Auto Fit timed out. Double-click a LiDAR point to retry.';
                if (result.reason === 'unavailable') hint = 'Auto Fit is unavailable. Try again later.';
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
