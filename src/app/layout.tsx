import type { Metadata } from "next";
import { Bricolage_Grotesque, Geist } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const geist = Geist({subsets:['latin'],variable:'--font-sans'});

const briGro = Bricolage_Grotesque({
  subsets: ["latin"],
});
// const briGro = Bricolage_Grotesque({
//   subsets: ["latin"],
// });



export const metadata: Metadata = {
  title: "RAG Chat-Bot",
  description: "A Chat Bot that doesn't suck.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={cn("h-full", "antialiased", briGro.className, "font-sans", geist.variable)}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
