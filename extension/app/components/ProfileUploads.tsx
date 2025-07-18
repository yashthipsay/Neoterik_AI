
import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { useSession } from "next-auth/react";

export function ProfileUploads({ userId, resumeInfo, githubUsername, onRefresh }: {
    userId: string,
    resumeInfo?: { filename: string, uploadedAt: string },
    githubUsername?: string,
    onRefresh: () => void
}) {
    const { data: session } = useSession();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [newResumeFile, setNewResumeFile] = useState<File | null>(null);
    const [newGithub, setNewGithub] = useState(githubUsername || "");
    const [githubLoading, setGithubLoading] = useState(false);
    const [githubSuccess, setGithubSuccess] = useState<string | null>(null);
    const [githubError, setGithubError] = useState<string | null>(null);

    const getAuthHeaders = () => {
      // NextAuth stores token under access_token or accessToken
      const token = session?.access_token || session?.accessToken;
      console.log("🔐 Using JWT for ProfileUploads:", token);
      return token
        ? { Authorization: `Bearer ${token}` }
        : {};
    };


    const handleUpdateResume = async () => {
        if (!newResumeFile) return;
        setLoading(true);
        setError(null);
        setSuccess(null);
        const formData = new FormData();
        formData.append('user_id', userId);
        formData.append('file', newResumeFile);
        try {
            const headers = getAuthHeaders();
            // Do not set Content-Type header manually when sending FormData
            const res = await fetch('http://localhost:8000/upload-resume', {
                method: 'POST',
                headers,  // only Authorization header is sent
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'Resume update failed');
            }
            setSuccess('Resume updated successfully!');
            setNewResumeFile(null);
            onRefresh();
        } catch (e: any) {
            setError(e.message || 'Failed to update resume');
        }
        setLoading(false);
    };

    const handleUpdateGithub = async () => {
        if (!newGithub) return;
        setGithubLoading(true);
        setGithubError(null);
        setGithubSuccess(null);
        const params = new URLSearchParams();
        params.append('user_id', userId);
        params.append('github_username', newGithub);
        try {
            const headers = getAuthHeaders();
            const res = await fetch('http://localhost:8000/submit-github', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...headers },
                body: params.toString(),
            });
            if (!res.ok) throw new Error('GitHub update failed');
            setGithubSuccess('GitHub username updated!');
            onRefresh();
        } catch (e: any) {
            setGithubError(e.message || 'Failed to update GitHub');
        }
        setGithubLoading(false);
    };

    return (
        <Card className="mb-6 animate-fade-in">
            <CardHeader>
                <CardTitle>Your Uploads</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="mb-4">
                    <div className="font-medium">Resume:</div>
                    <div className="flex flex-col gap-2">
                        {resumeInfo ? (
                            <div className="flex items-center gap-2">
                                <span>{resumeInfo.filename}</span>
                                <span className="text-xs text-gray-500">({resumeInfo.uploadedAt})</span>
                            </div>
                        ) : (
                            <span className="text-gray-500">No file has been uploaded.</span>
                        )}
                        <div className="flex items-center gap-2 mt-2">
                            <input
                                type="file"
                                accept="application/pdf"
                                onChange={e => setNewResumeFile(e.target.files?.[0] || null)}
                                className="border rounded px-2 py-1 text-sm"
                                disabled={loading}
                            />
                            <Button size="sm" variant="outline" onClick={handleUpdateResume} disabled={loading || !newResumeFile}>
                                {loading ? 'Uploading...' : 'Upload/Update'}
                            </Button>
                        </div>
                        {error && <div className="text-red-600 text-sm mt-2">{error}</div>}
                        {success && <div className="text-green-600 text-sm mt-2">{success}</div>}
                    </div>
                </div>
                <div className="mb-4">
                    <div className="font-medium">GitHub Username:</div>
                    <div className="flex items-center gap-2 mt-2">
                        <input
                            type="text"
                            value={newGithub}
                            onChange={e => setNewGithub(e.target.value)}
                            className="border rounded px-2 py-1 text-sm"
                            disabled={githubLoading}
                        />
                        <Button size="sm" variant="outline" onClick={handleUpdateGithub} disabled={githubLoading || !newGithub}>
                            {githubLoading ? 'Submitting...' : 'Submit'}
                        </Button>
                    </div>
                    {githubError && <div className="text-red-600 text-sm mt-2">{githubError}</div>}
                    {githubSuccess && <div className="text-green-600 text-sm mt-2">{githubSuccess}</div>}
                </div>
            </CardContent>
        </Card>
    );
}
