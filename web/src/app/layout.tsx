import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Yeafins — Play a Personalized Chess Engine",
  description:
    "Play against a policy model trained on Krishiv's chess games and guided by Stockfish evaluation.",
  applicationName: "Yeafins",
  openGraph: {
    title: "Yeafins — Play the moves Krishiv might make",
    description:
      "A personalized policy model and Stockfish hybrid, trained from 1,842 historical games.",
    type: "website",
    siteName: "Yeafins",
  },
  twitter: {
    card: "summary",
    title: "Yeafins — Personalized chess intelligence",
    description: "Play a chess engine trained to move like Krishiv.",
  },
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0d100f",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
