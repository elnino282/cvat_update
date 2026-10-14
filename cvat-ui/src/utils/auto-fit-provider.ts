// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import Axios from 'axios';
import { AutoFitProvider, FitRequest, FitResult } from 'cvat-canvas3d-wrapper';
import { getCVATStore } from 'cvat-store';

type FrameIdentity = Pick<FitRequest, 'taskId' | 'jobId' | 'frame'>;
type Prepared = FrameIdentity & { token?: string; abort: AbortController };
type Preparation = { frameToken: string; frame: number; state: 'preparing' | 'ready' | 'unavailable' };
const preparedFrames = new WeakMap<object, Prepared>();
const organizationParams = (): { org: string } => ({ org: getCVATStore().getState().organizations.current?.slug || '' });
const sameFrame = (a: FrameIdentity, b: FrameIdentity): boolean => (
    a.taskId === b.taskId && a.jobId === b.jobId && a.frame === b.frame
);

// Background preparation has a frame lifetime, independent of the selected drawing tool.
export function prepareRealAutoFitFrame(canvas: object, identity: FrameIdentity): () => void {
    const previous = preparedFrames.get(canvas);
    previous?.abort.abort();
    const entry: Prepared = { ...identity, abort: new AbortController() };
    preparedFrames.set(canvas, entry);
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    const current = (): boolean => !entry.abort.signal.aborted && preparedFrames.get(canvas) === entry;
    const poll = async (): Promise<void> => {
        if (!current()) return;
        try {
            const url = `/api/jobs/${identity.jobId}/auto-fit/prepare`;
            const config = {
                params: organizationParams(),
                signal: entry.abort.signal,
                timeout: 6000,
                'axios-retry': { retries: 0 },
            };
            const response = entry.token ? await Axios.get<Preparation>(url, {
                ...config, params: { ...config.params, ...identity, frameToken: entry.token },
            }) : await Axios.post<Preparation>(url, identity, config);
            if (!current() || response.data.frame !== identity.frame) return;
            entry.token = response.data.frameToken;
            failures = 0;
            if (response.data.state !== 'ready') {
                timer = setTimeout(poll, response.data.state === 'unavailable' ? 10000 : 500);
            }
        } catch (error) {
            if (!current()) return;
            if (Axios.isAxiosError(error) && error.response?.status === 410) entry.token = undefined;
            failures += 1;
            // Also recover a restarted service; never display background readiness in the canvas.
            timer = setTimeout(poll, Math.min(30000, 1000 * (2 ** Math.min(failures, 5))));
        }
    };
    poll();
    return () => {
        entry.abort.abort();
        if (timer) clearTimeout(timer);
        if (preparedFrames.get(canvas) === entry) preparedFrames.delete(canvas);
    };
}

export function createRealAutoFitProvider(kind: 'snap' | 'geometry-only', canvas?: object): AutoFitProvider {
    return {
        kind,
        async fit(request, signal): Promise<FitResult> {
            const url = `/api/jobs/${request.jobId}/auto-fit`;
            const params = organizationParams();
            const started = Date.now();
            const deadline = started + 1850;
            const cancel = (): void => {
                Axios.delete(`${url}/${request.requestId}`, { params, timeout: 1000 }).catch(() => {});
            };
            if (signal.aborted) throw new Error('Auto Fit cancelled');
            signal.addEventListener('abort', cancel, { once: true });
            try {
                const entry = canvas ? preparedFrames.get(canvas) : undefined;
                const payload = entry?.token && sameFrame(entry, request) ?
                    { ...request, frameToken: entry.token } : request;
                let response = await Axios.post<FitResult>(url, payload, {
                    params, signal, timeout: Math.max(1, deadline - Date.now()), 'axios-retry': { retries: 0 },
                });
                if (response.data.status === 'no-fit' && response.data.reason === 'frame-expired' &&
                    Date.now() < deadline - 400) {
                    // One recovery, within the original gesture deadline and request identity.
                    const prepared = await Axios.post<Preparation>(`${url}/prepare`, {
                        taskId: request.taskId, jobId: request.jobId, frame: request.frame,
                    }, {
                        params, signal, timeout: Math.max(1, deadline - Date.now()), 'axios-retry': { retries: 0 },
                    });
                    if (entry && sameFrame(entry, request)) entry.token = prepared.data.frameToken;
                    response = await Axios.post<FitResult>(url, { ...request, frameToken: prepared.data.frameToken }, {
                        params, signal, timeout: Math.max(1, deadline - Date.now()), 'axios-retry': { retries: 0 },
                    });
                }
                return response.data;
            } catch (error) {
                if (signal.aborted) throw error;
                const reason = Axios.isAxiosError(error) &&
                    (error.response?.status === 504 || error.code === 'ECONNABORTED') ? 'timeout' : 'unavailable';
                if (reason === 'timeout') cancel();
                return {
                    status: 'no-fit', requestId: request.requestId, frame: request.frame, provider: kind, reason,
                };
            } finally {
                signal.removeEventListener('abort', cancel);
            }
        },
    };
}
