import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  // gRPC client for NVIDIA Audio2Face-3D loads .proto files at runtime.
  serverExternalPackages: ["@grpc/grpc-js", "@grpc/proto-loader"],
  // Local LiteAvatar server (liteavatar-server/server.py, `npm run liteavatar`).
  rewrites() {
    return [{ source: "/liteavatar/:path*", destination: "http://127.0.0.1:8765/:path*" }];
  },
};

export default nextConfig;
