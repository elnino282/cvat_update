// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import { ArgumentError } from './exceptions';

export interface AutoFitProvenance {
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

export type AutoFitMetadata = AutoFitProvenance | Record<string, never>;

export const AUTO_FIT_METADATA_KEYS = [
    'version', 'request_id', 'frame', 'class_key', 'quality', 'fit_mode',
    'yaw_uncertain', 'prototype_key', 'prior_version', 'experimental',
] as const;

// A detached, validated record: editing geometry must not modify its creation provenance.
export function copyAutoFitMetadata(value: AutoFitMetadata = {}): AutoFitMetadata {
    const invalid = (): never => { throw new ArgumentError('Invalid Auto Fit creation provenance'); };
    if (!value || typeof value !== 'object' || Array.isArray(value)) invalid();
    const keys = Object.keys(value);
    if (!keys.length) return {};
    if (keys.length !== AUTO_FIT_METADATA_KEYS.length ||
        AUTO_FIT_METADATA_KEYS.some((key) => !Object.hasOwn(value, key))) invalid();
    if (value.version !== 1 || !Number.isInteger(value.frame) || value.frame < 0 ||
        value.frame > 2147483647) invalid();
    for (const key of ['request_id', 'class_key', 'prototype_key', 'prior_version'] as const) {
        const item = value[key];
        if (item === null && ['prototype_key', 'prior_version'].includes(key)) continue;
        if (typeof item !== 'string' || !item.length || item.length > 128 ||
            Array.from(item).some((char) => char.charCodeAt(0) < 32)) invalid();
    }
    if (!['confident', 'estimated'].includes(value.quality) ||
        !['geometry', 'hybrid_prior'].includes(value.fit_mode) ||
        typeof value.yaw_uncertain !== 'boolean' || typeof value.experimental !== 'boolean') invalid();
    return { ...value };
}
