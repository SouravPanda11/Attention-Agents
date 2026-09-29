import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Survey Benchmark v1",
  description: "Reproducible survey samples for evaluating web agents across increasing horizons.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
