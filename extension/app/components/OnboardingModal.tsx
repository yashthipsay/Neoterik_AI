import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { useSession } from "next-auth/react";
import { getToken } from "next-auth/jwt";

export function OnboardingModal({ onComplete, userId, onClose }: { onComplete: () => void, userId: string, onClose?: () => void }) {
    const { data: session } = useSession();
    const [resumeFile, setResumeFile] = useState<File | null>(null);
    const [githubUsername, setGithubUsername] = useState('');
    const [loading, setLoading] = useState(false);
    const [resumeUploaded, setResumeUploaded] = useState(false);
    const [githubSubmitted, setGithubSubmitted] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const getAuthHeaders = () => {
      // NextAuth stores it under either access_token or accessToken
      const token = session?.access_token || session?.accessToken;
      console.log("🔐 Uploading with JWT:", token);
      return token
        ? { Authorization: `Bearer ${token}` }
        : {};
    };

    const handleResumeUpload = async () => {
      if (!resumeFile) return;
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("file", resumeFile);

      try {
        const headers = getAuthHeaders();
        console.log("📤 resume upload headers:", headers);

        const res = await fetch("http://localhost:8000/upload-resume", {
          method: "POST",
          headers,
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Resume upload failed");
        }
        setResumeUploaded(true);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    const handleGithubSubmit = async () => {
      if (!githubUsername) return;
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      params.append("github_username", githubUsername);

      try {
        const headers = {
          ...getAuthHeaders(),
          "Content-Type": "application/x-www-form-urlencoded",
        };
        console.log("📤 github submit headers:", headers);

        const res = await fetch("http://localhost:8000/submit-github", {
          method: "POST",
          headers,
          body: params.toString(),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "GitHub username submit failed");
        }
        setGithubSubmitted(true);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
            <Card className="w-full max-w-md animate-fade-in relative">
                {/* Close button */}
                {onClose && (
                    <button
                        aria-label="Close"
                        onClick={onClose}
                        type="button"
                        className="relative right-4 ml-90 mt-2 top-4 z-10 flex items-center justify-center w-8 h-8 rounded-full bg-white shadow hover:bg-gray-100 border border-gray-200 text-gray-500 hover:text-gray-700 transition-all focus:outline-none"
                        style={{ lineHeight: 1, fontSize: '1.5rem', fontWeight: 700 }}
                    >
                        <span style={{ position: 'relative', top: '-1px' }}>&times;</span>
                    </button>
                )}
                <CardHeader>
                    <CardTitle>Welcome! Complete Your Profile</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="mb-6 text-center text-gray-600 text-sm">
                        To get started, upload your resume (required) and optionally add your GitHub username. This helps us personalize your cover letter experience!
                    </div>
                    <div className="mb-6">
                        <label className="block mb-2 font-medium">Upload Resume (PDF)</label>
                        <input
                            type="file"
                            accept="application/pdf"
                            onChange={e => setResumeFile(e.target.files?.[0] || null)}
                            className="mb-2 w-full border rounded px-2 py-1"
                            disabled={resumeUploaded}
                        />
                        <Button onClick={handleResumeUpload} disabled={!resumeFile || loading || resumeUploaded} className="w-full">
                            {loading && !resumeUploaded ? 'Uploading...' : resumeUploaded ? 'Uploaded!' : 'Upload Resume'}
                        </Button>
                    </div>
                    <div className="mb-6">
                        <label className="block mb-2 font-medium">GitHub Username (optional)</label>
                        <input
                            type="text"
                            value={githubUsername}
                            onChange={e => setGithubUsername(e.target.value)}
                            className="w-full border rounded px-2 py-1"
                            placeholder="e.g. johndoe"
                            disabled={githubSubmitted}
                        />
                        <Button onClick={handleGithubSubmit} disabled={!githubUsername || loading || githubSubmitted} className="w-full mt-2" variant="outline">
                            {loading && !githubSubmitted ? 'Submitting...' : githubSubmitted ? 'Submitted!' : 'Submit GitHub'}
                        </Button>
                    </div>
                    {error && <div className="text-red-600 text-sm mb-2">{error}</div>}
                    {resumeUploaded && (
                        <div className="text-green-600 text-sm mb-2">Resume parsed successfully! You’re ready to generate your cover letter.</div>
                    )}
                    <Button onClick={onComplete} disabled={!resumeUploaded} className="w-full mt-2">
                        Continue
                    </Button>
                </CardContent>
            </Card>
        </div>
    );
}
