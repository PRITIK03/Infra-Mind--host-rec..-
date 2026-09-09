import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // No rewrites needed — frontend calls NEXT_PUBLIC_API_URL directly.
  // App Router deploys to Vercel with zero config.
};

export default nextConfig;
