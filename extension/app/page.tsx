'use client'
import Image from 'next/image';
import { useSession, signOut } from "next-auth/react";
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function Home() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [isExtension, setIsExtension] = useState(false);


  useEffect(() => {
    if (typeof window !== "undefined" && window.location.protocol === "chrome-extension:") {
      setIsExtension(true);
    }
  }, []);

  const handleSignIn = () => {
    if(isExtension) {
      // Open login on localhost when in extension
      chrome.tabs.create({ url: "http://localhost:3000/auth/signin" });
    } else {
      router.push('/auth/signin');
    }
  };

    // --- Sign out handler ---
    const handleSignOut = () => {
      signOut(); // Call signOut from next-auth/react
    };
  return (
    <div className="extension-container scrollbar-thin">
    <div className="flex flex-col p-4">
      {/* --- Header Section --- */}
      <header className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200">
        <div className="flex items-center">
          <Image
            src="/Neoterik-Genesis.png"
            alt="Neoterik.ai Logo"
            width={40}
            height={40}
            priority
            className="mr-3"
          />
          <h1 className="text-xl font-semibold text-[#2D3047] dark:text-[#E5E7EB]">AI Cover Letter Assistant</h1>
        </div>

        {/* --- Dynamic Auth Button --- */}
        <div>
          {status === "loading" && (
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-500"></div>
          )}
          {status === "authenticated" && session && (
            <div className="flex items-center space-x-2">
               {session.user?.image && (
                 <Image
                   src={session.user.image}
                   alt="Profile"
                   width={24}
                   height={24}
                   className="rounded-full"
                 />
               )}
               <span className="text-sm font-medium hidden sm:inline">{session.user?.name?.split(' ')[0]}</span> {/* Show first name */}
              <button
                className="bg-red-500 hover:bg-red-600 text-white py-1 px-3 rounded-md text-sm font-medium transition-colors"
                onClick={handleSignOut}
              >
                Sign out
              </button>
            </div>
          )}
          {status === "unauthenticated" && (
            <button
              className="bg-blue-500 hover:bg-blue-600 text-white py-1 px-3 rounded-md text-sm font-medium transition-colors"
              onClick={handleSignIn}
            >
              Sign in
            </button>
          )}
        </div>
      </header>

      {/* --- Main Content (Only show if authenticated) --- */}
      {status === "authenticated" ? (
        <main className="flex-1">
          {/* Job Description Input */}
          <div className="bg-white dark:bg-[#374151] rounded-lg shadow-md p-5 mb-5">
            <h2 className="text-lg font-medium text-[#2D3047] dark:text-[#E5E7EB] mb-4">Generate Your Cover Letter</h2>
            <div className="mb-4">
              <label htmlFor="job-description" className="block mb-2 text-sm font-medium">
                Job Description
              </label>
              <textarea
                id="job-description"
                className="w-full p-3 border border-gray-200 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-md focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none"
                rows={4}
                placeholder="Paste the job description here..."
              />
            </div>
            {/* Resume Highlights Input */}
            <div className="mb-4">
              <label htmlFor="resume-highlights" className="block mb-2 text-sm font-medium">
                Key Resume Highlights (optional)
              </label>
              <textarea
                id="resume-highlights"
                className="w-full p-3 border border-gray-200 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-md focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none"
                rows={3}
                placeholder="List key points from your resume that you'd like to highlight..."
              />
            </div>
            <button className="bg-[#419D78] hover:bg-[#37876A] text-white py-2 px-4 rounded-md font-medium transition-colors">
              Generate Cover Letter
            </button>
          </div>

          {/* Recent Cover Letters Section */}
          <div className="bg-white dark:bg-[#374151] rounded-lg shadow-md p-5">
            <h2 className="text-lg font-medium text-[#2D3047] dark:text-[#E5E7EB] mb-4">Recent Cover Letters</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-4">Your recently generated cover letters will appear here.</p>
            <button className="bg-[#E0A458] hover:bg-[#C88F4B] text-white py-2 px-4 rounded-md font-medium transition-colors">
              View All
            </button>
          </div>
        </main>
      ) : (
        // --- Show this message if not authenticated ---
        <div className="flex-1 flex items-center justify-center text-center">
          <p className="text-gray-600 dark:text-gray-300">Please sign in to use the AI Cover Letter Assistant.</p>
        </div>
      )}

      {/* --- Footer --- */}
      <footer className="mt-6 text-center text-sm text-gray-500 dark:text-gray-400">
        &copy; 2025 Neoterik.ai | AI-powered cover letter assistant
      </footer>
    </div>
  </div>
  );
}
