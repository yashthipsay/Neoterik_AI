'use client'
import Image from 'next/image';
import { useSession, signOut } from "next-auth/react";
import { useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { CoverLetterGenerator } from '../components/CoverLetterGenerator';
import { CoverLetterPreview } from '../components/CoverLetterPreview';
import { UserDashboard } from '../components/UserDashboard';
import { OnboardingModal } from '../components/OnboardingModal';
import { ProfileUploads } from '../components/ProfileUploads';
import { Button } from '../components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
// import { StatusBadge } from './components/ui/StatusBadge';
import { fetchUserProfile } from '../lib/profile';

import { PlayCircle, Target, Zap, PenSquare } from 'lucide-react';

// Extend session user type to include id and github_username
type SessionUser = {
  id?: string;
  name?: string | null;
  email?: string | null;
  image?: string | null;
  github_username?: string;
};


export default function Home() {
  // --- App state and hooks ---
  const { data: session, status } = useSession();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('generate');
  const [generatedCoverLetter, setGeneratedCoverLetter] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  // --- Clean home page: no extension logic, no debug, no forced modal ---

  // --- Modal state (standalone, can be triggered from anywhere) ---
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingDismissed, setOnboardingDismissed] = useState(false);

  // --- Profile state (for dashboard/profile uploads) ---
  const [profile, setProfile] = useState<{ resumeInfo: { filename: string; uploadedAt: string } | null, githubUsername: string }>({ resumeInfo: null, githubUsername: '' });
  const [profileLoaded, setProfileLoaded] = useState(false);

  // --- Fetch profile (only in app, not extension) ---
  const refreshProfile = async () => {
    const user = session?.user as SessionUser | undefined;
    const userId = user?.id || user?.email || '';
    if (userId) {
      const data = await fetchUserProfile(userId);
      setProfile(data);
      setProfileLoaded(true);
    }
  };


  // --- Register user only once per session, right after authentication ---
  useEffect(() => {
    const user = session?.user as SessionUser | undefined;
    const userId = user?.id || user?.email || '';
    // Use sessionStorage to persist registration state for this session
    const registeredKey = `neoterik-registered-${userId}`;
    const alreadyRegistered = typeof window !== 'undefined' && userId && sessionStorage.getItem(registeredKey) === '1';
    if (status === "authenticated" && user && userId && !alreadyRegistered) {
      const payload = {
        id: userId,
        email: user?.email,
        name: user?.name,
        avatar_url: user?.image,
        github_username: user?.github_username || "",
      };
      fetch('http://localhost:8000/register-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text() || 'Failed to register user');
          if (typeof window !== 'undefined') sessionStorage.setItem(registeredKey, '1');
          // Only fetch profile once after registration
          refreshProfile();
        })
        .catch(() => { });
    }
  }, [status, session?.user]);

  // --- Show onboarding modal only once per session, only when profile is loaded and resume is missing ---
  useEffect(() => {
    if (
      status === 'authenticated' &&
      profileLoaded &&
      !profile.resumeInfo &&
      !onboardingDismissed
    ) {
      setShowOnboarding(true);
    }
  }, [status, profileLoaded, profile, onboardingDismissed]);

  const handleOnboardingClose = () => {
    setShowOnboarding(false);
    setOnboardingDismissed(true);
  };

  // --- Auth handlers ---
  const handleSignIn = () => {
    router.push('/auth/signin');
  };
  const handleSignOut = () => {
    signOut();
  };

  // --- Cover letter generation ---
  const handleGenerate = async (formData: any) => {
    setIsGenerating(true);
    setTimeout(() => {
      const mockCoverLetter = `Dear Hiring Manager,\n\nI am writing to express my strong interest in the ${formData.jobTitle} position at ${formData.companyName}. With my background in software development and passion for innovative technology solutions, I am excited about the opportunity to contribute to your team.\n\nIn my previous roles, I have successfully delivered high-quality software solutions that align perfectly with the requirements outlined in your job description. My experience includes:\n\n• Developing scalable web applications using modern frameworks\n• Collaborating with cross-functional teams to deliver projects on time\n• Implementing best practices for code quality and testing\n• Contributing to open-source projects and staying current with industry trends\n\n${formData.resumeHighlights ? `Key highlights from my experience include:\n${formData.resumeHighlights}\n\n` : ''}I am particularly drawn to ${formData.companyName} because of your commitment to innovation and excellence in the technology space. I believe my skills and enthusiasm would make me a valuable addition to your team.\n\n${formData.customInstructions ? `${formData.customInstructions}\n\n` : ''}Thank you for considering my application. I look forward to the opportunity to discuss how I can contribute to ${formData.companyName}'s continued success.\n\nBest regards,\n[Your Name]`;
      setGeneratedCoverLetter(mockCoverLetter);
      setActiveTab('preview');
      setIsGenerating(false);
    }, 3000);
  };

  // --- Tabs ---
  const tabs = [
    { id: 'generate', label: 'Generate', icon: '✨' },
    { id: 'preview', label: 'Preview', icon: '📄' },
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'library', label: 'Library', icon: '📚' }
  ];

  // --- User stats and activity (mock) ---
  const userStats = {
    coverLettersGenerated: 12,
    monthlyLimit: 50,
    successRate: 85,
    averageRating: 4.2
  };
  const recentActivity = [
    { title: 'Software Engineer', company: 'Google', status: 'success' as const, statusText: 'Applied', date: '2 days ago' },
    { title: 'Product Manager', company: 'Microsoft', status: 'pending' as const, statusText: 'In Review', date: '5 days ago' },
    { title: 'Data Scientist', company: 'Netflix', status: 'warning' as const, statusText: 'Follow-up', date: '1 week ago' }
  ];

  // --- Render ---
  return (
    // The new stage: a deep, sophisticated, and professional canvas.
    <div className="min-h-screen bg-[#111111] text-gray-300 font-sans">
      
      {/* ACT I: THE HERO
        This is no longer a split screen. It is a cinematic reveal.
        Centered, focused, and built around your two stars: the Logo and the Video.
      */}
      <section className="relative text-center py-24 sm:py-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Subtle background glow for atmosphere. It's lighting, not decoration. */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[60%] bg-gradient-to-t from-[#419D78]/10 to-transparent blur-3xl -z-0"></div>

        <div className="max-w-4xl mx-auto relative z-10">
          {/* THE STAR: Your logo. Given the prominence it deserves. */}
          <Image
            src="/Neoterik-Genesis.png" // ASSUMPTION: You have a sleek, monochrome or brand-colored SVG logo.
            alt="Neoterik.ai Logo"
            width={96} // Larger, more confident.
            height={96}
            className="mx-auto mb-8"
          />

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-50 tracking-tight">
            Don't Just Apply. 
            <span className="bg-gradient-to-r from-[#419D78] to-[#6ddaa8] bg-clip-text text-transparent block mt-2">
              Arrive.
            </span>
          </h1>

          <p className="mt-6 max-w-2xl mx-auto text-lg sm:text-xl text-gray-400 leading-8">
            Neoterik.ai crafts bespoke cover letters that open doors. Our AI understands nuance, tone, and strategy, ensuring your first impression is your best.
          </p>

          {/* THE CO-STAR: The Demo Video. An irresistible call to action. */}
          <div className="mt-12">
            <div className="relative group w-full max-w-2xl mx-auto rounded-xl shadow-2xl shadow-[#419D78]/20 overflow-hidden cursor-pointer">
              <Image
                src="/video-thumbnail.jpg" // A compelling thumbnail from your video.
                alt="Neoterik.ai Demo Video Thumbnail"
                width={1920}
                height={1080}
                className="w-full h-auto transform group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-black/40 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
                <PlayCircle className="text-white w-20 h-20 transform group-hover:scale-110 transition-transform duration-300" />
              </div>
            </div>
          </div>
          
          <div className="mt-10 flex justify-center gap-4">
            <Button size="lg" className="bg-[#419D78] text-white hover:bg-[#37876A] text-lg px-8">
              Install for Chrome
            </Button>
          </div>
        </div>
      </section>

      {/* ACT II: THE METHOD
        Elevated from simple text to a structured, professional showcase.
        No emojis. We use clean, intentional iconography.
      */}
      <section className="py-20 sm:py-24 bg-black/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-100">A Symphony in Three Steps</h2>
            <p className="mt-4 text-lg text-gray-400">
              From job post to polished draft, seamlessly.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: <Target className="w-10 h-10 text-[#E0A458]" />, // Ochre for accent
                title: "Pinpoint the Target",
                description: "Our extension seamlessly integrates with job boards, instantly analyzing the role, company, and requirements."
              },
              {
                icon: <Zap className="w-10 h-10 text-[#E0A458]" />,
                title: "Generate with Intent",
                description: "With one click, our AI composes a first draft, strategically aligning your profile with the job's key needs."
              },
              {
                icon: <PenSquare className="w-10 h-10 text-[#E0A458]" />,
                title: "Refine to Perfection",
                description: "You are the final editor. Easily customize, add personal anecdotes, and perfect the tone."
              }
            ].map((feature, i) => (
              <div key={i} className="bg-[#1a1a1a] p-8 rounded-lg border border-gray-800 hover:border-[#419D78] transition-colors duration-300">
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-gray-100 mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ACT III: THE INVITATION (CTA)
        Direct, confident, and irresistible.
      */}
      <section className="py-20 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-50 mb-6">
            Your Next Chapter Awaits.
          </h2>
          <p className="text-xl mb-8 text-gray-400">
            Stop writing cover letters. Start opening doors. Install Neoterik.ai and transform your job search today.
          </p>
          <Button  
            size="lg"  
            variant="secondary"
            className="text-lg px-12 py-6 bg-gradient-to-r from-[#419D78] to-[#6ddaa8] text-white font-bold hover:opacity-90 transition-opacity"
          >
            Get Started for Free
          </Button>
        </div>
      </section>
      
      {/* EPILOGUE: The Footer
        Every professional production needs credits.
      */}
      <footer className="border-t border-gray-800 py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-500">
              <p>&copy; {new Date().getFullYear()} Neoterik.ai. All rights reserved.</p>
              {/* Add links to Privacy Policy, Terms of Service, etc. here */}
          </div>
      </footer>
    </div>
  );
}