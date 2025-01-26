
/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    async rewrites() {
      return [
        {
          source: '/uv-index',
          destination: '/uv-index.js',
        },
      ];
    },
    // Add this to handle static HTML
    async headers() {
      return [
        {
          source: '/:path*',
          headers: [
            {
              key: 'Cache-Control',
              value: 'no-store',
            },
          ],
        },
      ];
    },
  }
  
  module.exports = nextConfig 
  