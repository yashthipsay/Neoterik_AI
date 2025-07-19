import './globals.css';
import { Metadata } from 'next';
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: 'Neoterik.ai Cover Letter Assistant',
  description: 'AI-powered cover letter generator that helps you craft personalized cover letters based on job descriptions.',
  icons: {
    icon: '/icons/icon16.png',
  }
};

// This is the true root layout - keep it minimal.
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}