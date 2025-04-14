import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Configure assetPrefix for static export compatibility with Chrome extensions
  assetPrefix: process.env.NODE_ENV === 'production' ? '.' : undefined,
  // Enable static export
  output: 'export',
  // Ensure trailing slashes for static file routing
  trailingSlash: true,
};

export default nextConfig;
