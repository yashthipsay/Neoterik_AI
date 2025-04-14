import './globals.css';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Neoterik.ai Cover Letter Assistant',
  description: 'AI-powered cover letter generator that helps you craft personalized cover letters based on job descriptions.',
  icons: {
    icon: '/icons/icon16.png',
  }
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
