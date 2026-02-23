import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/survey_v1/q1_a.jpg", destination: "/Basketball.jpg" },
      { source: "/survey_v1/q1_b.jpg", destination: "/Football.jpg" },
      { source: "/survey_v1/q2_a.webp", destination: "/Gymnastics.webp" },
      { source: "/survey_v1/q2_b.jpg", destination: "/Cycling.jpg" },
      { source: "/survey_v1/q3_a.jpg", destination: "/Badminton.jpg" },
      { source: "/survey_v1/q3_b.jpg", destination: "/Tennis.jpg" },
      { source: "/survey_v1/q4_a.webp", destination: "/Rowing.webp" },
      { source: "/survey_v1/q4_b.jpg", destination: "/Boat.jpg" },
      { source: "/survey_v1/attn_a.jpg", destination: "/Soccer.jpg" },
      { source: "/survey_v1/attn_b.jpeg", destination: "/golf.jpeg" },
    ];
  },
};

export default nextConfig;
