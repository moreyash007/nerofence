/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static export: the viewer is 100% client-side, so Vercel serves plain
  // files with no serverless function, no cold start and no bundle-size limit.
  // This is also what keeps the security posture intact -- a report dropped into
  // the page is parsed in the browser and never uploaded anywhere.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
