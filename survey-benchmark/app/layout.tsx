import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Survey Benchmark v0",
  description: "Fixed survey workflows for evaluating web agents.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
