import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Yeafins — Play a Personalized Chess Engine",
  description:
    "Play against a chess engine trained on Krishiv's games.",
  applicationName: "Yeafins",
  openGraph: {
    title: "Yeafins — Play the moves Krishiv might make",
    description:
      "Play against a personalized chess engine trained on Krishiv's games.",
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
