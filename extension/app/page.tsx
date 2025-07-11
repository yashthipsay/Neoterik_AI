'use client'
import Image from 'next/image';
import { useSession, signOut } from "next-auth/react";
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { CoverLetterGenerator } from './components/CoverLetterGenerator';
import { CoverLetterPreview } from './components/CoverLetterPreview';
import { UserDashboard } from './components/UserDashboard';
import { Button } from './components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/Card';
import { StatusBadge } from './components/ui/StatusBadge';

export default function Home() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [isExtension, setIsExtension] = useState(false);
  const [activeTab, setActiveTab] = useState('generate');
  const [generatedCoverLetter, setGeneratedCoverLetter] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    if (status === "authenticated" && session?.user?.id) {
      fetch("http://localhost:8000/register-user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: session.user.id,
          email: session.user.email,
          name: session.user.name,
          avatar_url: session.user.image,
          github_username: session.user.github_username || "",
        }),
      }).catch((err) => {
        console.error("Failed to register user:", err);
      });
    }
  }, [status, session?.user?.id]);

  // Mock user stats
  const userStats = {
    coverLettersGenerated: 12,
    monthlyLimit: 50,
    successRate: 85,
    averageRating: 4.2
  };

  const recentActivity = [
    {
      title: 'Software Engineer',
      company: 'Google',
      status: 'success' as const,
      statusText: 'Applied',
      date: '2 days ago'
    },
    {
      title: 'Product Manager',
      company: 'Microsoft',
      status: 'pending' as const,
      statusText: 'In Review',
      date: '5 days ago'
    },
    {
      title: 'Data Scientist',
      company: 'Netflix',
      status: 'warning' as const,
      statusText: 'Follow-up',
      date: '1 week ago'
    }
  ];

  useEffect(() => {
    if (typeof window !== "undefined" && window.location.protocol === "chrome-extension:") {
      setIsExtension(true);
    }
  }, []);

  const handleSignIn = () => {
    if(isExtension) {
      const width = 600;
      const height = 700;
      const left = Math.round((window.screen.width - width) / 2);
      const top = Math.round((window.screen.height - height) / 2);

      chrome.windows.create({
        url: "http://localhost:3000/auth/signin",
        type: "popup",
        width: width,
        height: height,
        left: left,
        top: top
      });
    } else {
      router.push('/auth/signin');
    }
  };

  const handleSignOut = () => {
    signOut();
  };

  const handleGenerate = async (formData: any) => {
    setIsGenerating(true);
    
    // Simulate API call
    setTimeout(() => {
      const mockCoverLetter = `Dear Hiring Manager,

I am writing to express my strong interest in the ${formData.jobTitle} position at ${formData.companyName}. With my background in software development and passion for innovative technology solutions, I am excited about the opportunity to contribute to your team.

In my previous roles, I have successfully delivered high-quality software solutions that align perfectly with the requirements outlined in your job description. My experience includes:

• Developing scalable web applications using modern frameworks
• Collaborating with cross-functional teams to deliver projects on time
• Implementing best practices for code quality and testing
• Contributing to open-source projects and staying current with industry trends

${formData.resumeHighlights ? `Key highlights from my experience include:\n${formData.resumeHighlights}\n\n` : ''}

I am particularly drawn to ${formData.companyName} because of your commitment to innovation and excellence in the technology space. I believe my skills and enthusiasm would make me a valuable addition to your team.

${formData.customInstructions ? `${formData.customInstructions}\n\n` : ''}

Thank you for considering my application. I look forward to the opportunity to discuss how I can contribute to ${formData.companyName}'s continued success.

Best regards,
[Your Name]`;

      setGeneratedCoverLetter(mockCoverLetter);
      setActiveTab('preview');
      setIsGenerating(false);
    }, 3000);
  };

  const tabs = [
    { id: 'generate', label: 'Generate', icon: '✨' },
    { id: 'preview', label: 'Preview', icon: '📄' },
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'library', label: 'Library', icon: '📚' }
  ];

  return (
    <div className="extension-container scrollbar-thin">
      <div className="flex flex-col h-full">
        {/* Header Section */}
        <header className="flex items-center justify-between p-4 border-b border-gray-200 bg-white sticky top-0 z-10">
          <div className="flex items-center gap-3">
            <Image
              src="/Neoterik-Genesis.png"
              alt="Neoterik.ai Logo"
              width={32}
              height={32}
              priority
              className="rounded-lg"
            />
            <div>
              <h1 className="text-lg font-bold text-[#2D3047]">Neoterik.ai</h1>
              <p className="text-xs text-gray-500">AI Cover Letter Assistant</p>
            </div>
          </div>

          {/* Auth Section */}
          <div className="flex items-center gap-2">
            {status === "loading" && (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-500"></div>
            )}
            {status === "authenticated" && session && (
              <div className="flex items-center gap-2">
                {session.user?.image && (
                  <Image
                    src={session.user.image}
                    alt="Profile"
                    width={24}
                    height={24}
                    className="rounded-full"
                  />
                )}
                <div className="text-right">
                  <div className="text-xs font-medium">{session.user?.name?.split(' ')[0]}</div>
                  <StatusBadge status="success" className="text-xs">Pro</StatusBadge>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleSignOut}
                  className="text-xs"
                >
                  Sign out
                </Button>
              </div>
            )}
            {status === "unauthenticated" && (
              <Button
                size="sm"
                onClick={handleSignIn}
              >
                Sign in
              </Button>
            )}
          </div>
        </header>

        {/* Main Content */}
        {status === "authenticated" ? (
          <div className="flex-1 flex flex-col">
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200 bg-white px-4">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-[#419D78] text-[#419D78]'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <span>{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-auto p-4">
              {activeTab === 'generate' && (
                <CoverLetterGenerator
                  onGenerate={handleGenerate}
                  isGenerating={isGenerating}
                />
              )}
              
              {activeTab === 'preview' && generatedCoverLetter && (
                <CoverLetterPreview
                  coverLetter={generatedCoverLetter}
                  onEdit={() => setActiveTab('generate')}
                  onSave={() => {
                    // Handle save logic
                    console.log('Saving cover letter...');
                  }}
                  onCopy={() => {
                    console.log('Cover letter copied!');
                  }}
                />
              )}
              
              {activeTab === 'preview' && !generatedCoverLetter && (
                <Card className="text-center py-12">
                  <CardContent>
                    <div className="text-6xl mb-4">📄</div>
                    <h3 className="text-lg font-semibold mb-2">No Cover Letter Yet</h3>
                    <p className="text-gray-600 mb-4">Generate your first cover letter to see the preview here.</p>
                    <Button onClick={() => setActiveTab('generate')}>
                      Start Generating
                    </Button>
                  </CardContent>
                </Card>
              )}
              
              {activeTab === 'dashboard' && (
                <UserDashboard stats={userStats} recentActivity={recentActivity} />
              )}
              
              {activeTab === 'library' && (
                <Card className="text-center py-12">
                  <CardContent>
                    <div className="text-6xl mb-4">📚</div>
                    <h3 className="text-lg font-semibold mb-2">Your Cover Letter Library</h3>
                    <p className="text-gray-600 mb-4">Save and organize your generated cover letters here.</p>
                    <Button variant="outline">
                      Coming Soon
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        ) : (
          /* Unauthenticated State */
          <div className="flex-1 flex items-center justify-center p-4">
            <Card className="text-center max-w-sm">
              <CardContent className="py-8">
                <div className="text-6xl mb-4">🚀</div>
                <h3 className="text-lg font-semibold mb-2">Welcome to Neoterik.ai</h3>
                <p className="text-gray-600 mb-6">
                  Create personalized, AI-powered cover letters that help you stand out from the competition.
                </p>
                <Button onClick={handleSignIn} className="w-full">
                  Get Started
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Footer */}
        <footer className="p-3 text-center text-xs text-gray-500 border-t border-gray-200 bg-white">
          &copy; 2025 Neoterik.ai | AI-powered career tools
        </footer>
      </div>
    </div>
  );
}