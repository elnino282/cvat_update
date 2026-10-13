// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

export type Triple = [number, number, number];

export interface FitIdentity {
    taskId: number;
    jobId: number;
    frame: number;
    labelId: number;
    objectType: 'shape' | 'track';
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
}

export type FitResult = {
    status: 'ok';
    requestId: string;
    frame: number;
    center: Triple;
    size: Triple;
    rotation: Triple;
    provider: 'mock' | 'snap';
    confidence?: number;
} | {
    status: 'no-fit';
    requestId: string;
    frame: number;
    reason: 'no-points' | 'ambiguous' | 'insufficient-evidence' | 'unavailable';
};

export interface AutoFitProvider {
    readonly kind: 'mock' | 'snap';
    fit(request: FitRequest, signal: AbortSignal): Promise<FitResult>;
}

export interface AutoFitDraw {
    identity: FitIdentity;
    provider: AutoFitProvider;
    isCurrent(): boolean;
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
    if (result.status !== 'ok' || result.provider !== provider ||
        !['mock', 'snap'].includes(result.provider) ||
        !finiteTriple(result.center) || !finiteTriple(result.rotation) || !finiteTriple(result.size) ||
        result.size.some((n) => n <= 0) ||
        (result.confidence !== undefined && (!Number.isFinite(result.confidence) ||
            result.confidence < 0 || result.confidence > 1))) {
        throw new Error('Auto Fit returned invalid cuboid geometry');
    }
    return [...result.center, ...result.rotation, ...result.size, 0, 0, 0, 0, 0, 0, 0];
}

export type FitOutcome = { status: 'ok'; request: FitRequest; points: number[] } |
{ status: 'retry'; hint: string } | { status: 'cancelled' };

let nextRequest = 0;

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

    public async start(draw: AutoFitDraw, seed: FitSeed, timeout = 10000): Promise<FitOutcome> {
        if (this.pending || !draw.isCurrent()) return { status: 'cancelled' };
        if (![seed.x, seed.y, seed.z].every(Number.isFinite) ||
            !Number.isInteger(seed.pointIndex) || seed.pointIndex < 0) {
            return { status: 'retry', hint: 'Double-click a visible LiDAR point to retry.' };
        }
        const request: FitRequest = {
            ...draw.identity,
            seed: { ...seed },
            requestId: `autofit-${++nextRequest}`,
            requestedAt: Date.now(),
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
                return {
                    status: 'retry',
                    hint: result.reason === 'unavailable' ? 'Auto Fit is unavailable. Try again later.' :
                        'No confident fit. Double-click another LiDAR point to retry.',
                };
            }
            return { status: 'ok', request, points: fitResultToPoints(result, request, draw.provider.kind) };
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
