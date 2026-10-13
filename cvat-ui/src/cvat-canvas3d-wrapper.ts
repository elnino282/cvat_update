// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import {
    Canvas3d,
    MouseInteraction,
    ViewType,
    CameraAction,
    ViewsDOM,
    CanvasMode,
} from 'cvat-canvas3d/src/typescript/canvas3d';
import { OrientationVisibility } from 'cvat-canvas3d/src/typescript/canvas3dModel';

export {
    Canvas3d, MouseInteraction, ViewType, CameraAction, CanvasMode,
};

export type { ViewsDOM, OrientationVisibility };
export type { AutoFitDraw, FitRequest } from 'cvat-canvas3d/src/typescript/autoFit';
export { createMockAutoFitProvider } from 'cvat-canvas3d/src/typescript/autoFitMock';
export type { MockMode } from 'cvat-canvas3d/src/typescript/autoFitMock';
