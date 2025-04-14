import Image from 'next/image';

export default function Home() {
  return (
    <div className="extension-container scrollbar-thin">
      <div className="flex flex-col p-4">
        <header className="flex items-center mb-6 pb-4 border-b border-gray-200">
          <Image
            src="/Neoterik-Genesis.png"
            alt="Neoterik.ai Logo"
            width={40}
            height={40}
            priority
            className="mr-3"
          />
          <h1 className="text-xl font-semibold text-[#2D3047] dark:text-[#E5E7EB]">AI Cover Letter Assistant</h1>
        </header>

        <main className="flex-1">
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

          <div className="bg-white dark:bg-[#374151] rounded-lg shadow-md p-5">
            <h2 className="text-lg font-medium text-[#2D3047] dark:text-[#E5E7EB] mb-4">Recent Cover Letters</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-4">Your recently generated cover letters will appear here.</p>
            <button className="bg-[#E0A458] hover:bg-[#C88F4B] text-white py-2 px-4 rounded-md font-medium transition-colors">
              View All
            </button>
          </div>
        </main>

        <footer className="mt-6 text-center text-sm text-gray-500 dark:text-gray-400">
          &copy; 2025 Neoterik.ai | AI-powered cover letter assistant
        </footer>
      </div>
    </div>
  );
}
