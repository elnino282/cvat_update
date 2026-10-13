// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import { AutoFitProvider, FitRequest, FitResult } from './autoFit';

export type MockMode = 'success' | 'no-fit' | 'reject' | 'timeout' | 'nan' | 'infinity' |
    'negative' | 'wrong-frame' | 'wrong-request' | 'wrong-provider';

// Fixture dimensions in metres, with zero Euler rotation. This provider does not segment or fit a car.
export function createMockAutoFitProvider(mode: MockMode = 'success', delay = 300): AutoFitProvider {
    return {
        kind: 'mock',
        fit(request: FitRequest, signal: AbortSignal): Promise<FitResult> {
            return new Promise((resolve, reject) => {
                if (signal.aborted) {
                    reject(new Error('Cancelled'));
                    return;
                }
                let timer: ReturnType<typeof setTimeout>;
                const onAbort = (): void => {
                    clearTimeout(timer);
                    reject(new Error('Cancelled'));
                };
                timer = setTimeout(() => {
                    signal.removeEventListener('abort', onAbort);
                    if (mode === 'reject') {
                        reject(new Error('Mock rejection'));
                    } else if (mode === 'no-fit') {
                        resolve({
                            status: 'no-fit', requestId: request.requestId, frame: request.frame, reason: 'ambiguous',
                        });
                    } else {
                        resolve({
                            status: 'ok',
                            requestId: mode === 'wrong-request' ? 'mismatched' : request.requestId,
                            frame: request.frame + (mode === 'wrong-frame' ? 1 : 0),
                            center: [mode === 'nan' ? NaN : request.seed.x, request.seed.y, request.seed.z],
                            size: [mode === 'negative' ? -4 : 4, 1.8, 1.6],
                            rotation: [0, 0, mode === 'infinity' ? Infinity : 0],
                            provider: mode === 'wrong-provider' ? 'snap' : 'mock',
                        });
                    }
                }, mode === 'timeout' ? 60000 : Math.min(60000, Math.max(0, delay)));
                signal.addEventListener('abort', onAbort, { once: true });
            });
        },
    };
}
