// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

// Local browser-test frontend for an existing CVAT stack. No backend container reconfiguration.
const base = require('../cvat-ui/webpack.config.js');

module.exports = (env, argv) => {
    const config = base(env, argv);
    config.devServer.hot = false;
    config.devServer.liveReload = false;
    config.devServer.static.watch = false;
    const backendOrigin = new URL(env.API_URL).origin;
    const frontendOrigin = `http://${config.devServer.host}:${config.devServer.port}`;
    const proxy = config.devServer.proxy[0];
    proxy.onProxyReq = (request, original) => {
        // Only rewrite the origin sent by our own development frontend.
        if (original.headers.origin === frontendOrigin) request.setHeader('Origin', backendOrigin);
        request.setHeader('X-FORWARDED-HOST', new URL(frontendOrigin).host);
    };
    proxy.onProxyRes = (response) => {
        // Traefik can restore the backend host in Tus Location headers. Keep browser uploads same-origin.
        if (response.headers.location?.startsWith(`${backendOrigin}/`)) {
            response.headers.location = frontendOrigin + response.headers.location.slice(backendOrigin.length);
        }
    };
    return config;
};
