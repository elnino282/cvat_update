// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import * as THREE from 'three';
import { FitSeed } from './autoFit';

// Point-cloud-only raycast. Intersection.point is on the ray, so read the actual vertex instead.
// Calibrate the world tolerance to a six CSS pixel radius and rendered point size at cloud distance,
// then screen-filter each candidate at its actual depth. This prevents far-cloud tolerance accepting empty space.
export function pickAutoFitSeed(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
    ndc: THREE.Vector2,
    width: number,
    height: number,
): FitSeed | null {
    if (!width || !height || ![ndc.x, ndc.y].every(Number.isFinite) ||
        Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1) return null;
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    const clouds = scene.children.filter((object): object is THREE.Points => object instanceof THREE.Points);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    let best: { seed: FitSeed; distance: number } | null = null;
    const vertex = new THREE.Vector3();
    const projected = new THREE.Vector3();
    for (const cloud of clouds) {
        if (!cloud.visible) continue;
        const positions = cloud.geometry.getAttribute('position');
        if (!positions?.count) continue;
        if (!cloud.geometry.boundingSphere) cloud.geometry.computeBoundingSphere();
        const sphere = cloud.geometry.boundingSphere.clone().applyMatrix4(cloud.matrixWorld);
        const maxDistance = camera.position.distanceTo(sphere.center) + sphere.radius;
        const pixelWorld = camera instanceof THREE.PerspectiveCamera ?
            (2 * maxDistance * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / (height * camera.zoom) :
            (camera.top - camera.bottom) / (height * camera.zoom);
        const material = cloud.material as THREE.PointsMaterial;
        ray.params.Points.threshold = Math.max(pixelWorld * 6, material.size / 2);
        for (const hit of ray.intersectObject(cloud, false)) {
            if (!Number.isInteger(hit.index) || hit.index < 0 || hit.index >= positions.count) continue;
            vertex.fromBufferAttribute(positions, hit.index);
            if (![vertex.x, vertex.y, vertex.z].every(Number.isFinite)) continue;
            vertex.applyMatrix4(cloud.matrixWorld);
            projected.copy(vertex).project(camera);
            if (projected.z < -1 || projected.z > 1) continue;
            const pixels = Math.hypot(((projected.x - ndc.x) * width) / 2, ((projected.y - ndc.y) * height) / 2);
            const distance = camera.position.distanceTo(vertex);
            const halfFov = camera instanceof THREE.PerspectiveCamera ? camera.fov / 2 : 0;
            const viewSpan = 2 * distance * Math.tan(THREE.MathUtils.degToRad(halfFov));
            const localPixelWorld = camera instanceof THREE.PerspectiveCamera ?
                viewSpan / (height * camera.zoom) : pixelWorld;
            const radius = Math.max(
                6, material.sizeAttenuation ? material.size / (2 * localPixelWorld) : material.size / 2,
            );
            if (pixels <= radius && (!best || distance < best.distance)) {
                best = {
                    seed: {
                        x: vertex.x, y: vertex.y, z: vertex.z, pointIndex: hit.index,
                    },
                    distance,
                };
            }
        }
    }
    return best?.seed || null;
}
