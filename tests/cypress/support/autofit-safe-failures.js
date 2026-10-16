// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

// Allure serializes assertion actual/expected without a depth limit. Canvas DOM
// subjects can contain the Three.js scene and React graph, so bound only the
// diagnostic values before Allure's fail listener receives the original error.
function diagnosticValue(value, depth = 0, seen = new WeakSet(), budget = { remaining: 128 }) {
    if (typeof value === 'string') return value.length > 1024 ? `${value.slice(0, 1024)}…` : value;
    if (typeof value === 'bigint') return String(value);
    if (typeof value === 'function') return '[Function]';
    if (value === null || typeof value !== 'object') return value;
    if (Cypress.dom.isDom(value)) {
        const element = value.jquery ? value[0] : value;
        return `[DOM ${String(element?.nodeName || 'collection').slice(0, 80)}]`;
    }
    if (seen.has(value)) return '[Circular]';
    if (depth >= 3 || budget.remaining <= 0) return '[Truncated]';
    seen.add(value);
    const remainingBudget = budget;
    remainingBudget.remaining -= 1;
    if (Array.isArray(value)) {
        const result = value.slice(0, 20).map((item) => diagnosticValue(item, depth + 1, seen, budget));
        if (value.length > 20) result.push(`[${value.length - 20} more items]`);
        return result;
    }
    const result = {};
    const keys = Object.keys(value);
    keys.slice(0, 20).forEach((key) => {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        Object.defineProperty(result, key, {
            value: descriptor && 'value' in descriptor ?
                diagnosticValue(descriptor.value, depth + 1, seen, budget) : '[Accessor]',
            enumerable: true,
        });
    });
    return result;
}

Cypress.on('fail', (error) => {
    [error, error.matcherResult].filter(Boolean).forEach((details) => {
        ['actual', 'expected'].forEach((key) => {
            if (key in details) Object.assign(details, { [key]: diagnosticValue(details[key]) });
        });
    });
    // Keep message, stack, identity and the failure intact. Allure's listener,
    // registered next, reports the error and rethrows it to Cypress.
});
