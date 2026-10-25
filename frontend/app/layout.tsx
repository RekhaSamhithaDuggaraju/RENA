import "./globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = { title: "RENA — Unlearning Audit Lab", description: "Interactive research workspace for independent machine unlearning verification." };
export default function RootLayout({ children }: { children: React.ReactNode }) { return <html lang="en"><body className="antialiased">{children}</body></html>; }
