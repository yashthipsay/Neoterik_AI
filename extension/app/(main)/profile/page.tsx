'use client'
import { useSession } from "next-auth/react";
import { useEffect, useState } from "react";
import Image from "next/image";
import { Button } from "../../components/ui/Button";
import { Card, CardContent } from "../../components/ui/Card";
import { fetchUserProfile } from "../../lib/profile";
import { ProfileUploads } from "../../components/ProfileUploads";

type ProfileData = {
    resumeInfo: any;
    githubUsername: string;
    recentCoverLetters: any[];
};

export default function ProfilePage() {
    const { data: session, status } = useSession();
    const [profile, setProfile] = useState<ProfileData>({ resumeInfo: null, githubUsername: '', recentCoverLetters: [] });
    const user = session?.user as { id?: string; name?: string | null; email?: string | null; image?: string | null };
    useEffect(() => {
        if (user?.id) {
            fetchUserProfile(user.id).then(data => setProfile(data as ProfileData));
        }
    }, [user?.id]);

    if (status !== "authenticated") {
        return (
            <div className="min-h-screen bg-[#111111] text-gray-300 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-100 mb-4">Access Restricted</h1>
                    <p className="text-gray-400">Please sign in to view your profile.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#111111] text-gray-300">
            <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-100 mb-2">Your Profile</h1>
                    <p className="text-gray-400">Manage your account settings and view your activity</p>
                </div>
                
                <Card className="bg-[#1a1a1a] border border-gray-800">
                    <CardContent className="p-8">
                        <div className="flex items-center gap-6 mb-8">
                            <Image 
                                src={user.image || "/default-avatar.png"} 
                                alt="Profile" 
                                width={80} 
                                height={80} 
                                className="rounded-full border-2 border-gray-700" 
                            />
                            <div>
                                <div className="text-2xl font-bold text-gray-100">{user.name}</div>
                                <div className="text-gray-400 mb-2">{user.email}</div>
                                <div className="text-sm text-[#419D78] bg-[#419D78]/20 px-3 py-1 rounded-full inline-block">
                                    Plan: Pro
                                </div>
                            </div>
                        </div>
                        
                        {/* Profile Uploads Section */}
                        <div className="mb-8">
                            <ProfileUploads
                                userId={user.id || user.email || ''}
                                resumeInfo={profile.resumeInfo || undefined}
                                githubUsername={profile.githubUsername}
                                onRefresh={() => {
                                    if (user.id) fetchUserProfile(user.id).then(data => setProfile(data as ProfileData));
                                }}
                            />
                        </div>
                        
                        {/* GitHub Username Section */}
                        <div className="mb-8">
                            <h3 className="text-lg font-semibold text-gray-100 mb-4">GitHub Integration</h3>
                            {profile.githubUsername ? (
                                <div className="bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
                                    <div className="text-sm text-gray-400 mb-1">Connected GitHub Username</div>
                                    <div className="text-gray-200 font-medium">{profile.githubUsername}</div>
                                </div>
                            ) : (
                                <div className="text-gray-500 bg-gray-800/50 rounded-lg px-4 py-3 border border-gray-700">
                                    No GitHub username connected yet.
                                </div>
                            )}
                        </div>
                        
                        {/* Recent Cover Letters Section */}
                        <div>
                            <h3 className="text-lg font-semibold text-gray-100 mb-4">Recent Cover Letters</h3>
                            {(Array.isArray(profile.recentCoverLetters) && profile.recentCoverLetters.length > 0) ? (
                                <div className="space-y-3">
                                    {profile.recentCoverLetters.map((cl, idx) => (
                                        <div key={idx} className="bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
                                            <div className="text-gray-200 font-medium">{cl.title}</div>
                                            <div className="text-sm text-gray-400">{cl.date}</div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-gray-500 bg-gray-800/50 rounded-lg px-4 py-3 border border-gray-700 text-center">
                                    <div className="text-gray-400 mb-2">No cover letters generated yet.</div>
                                    <div className="text-sm text-gray-500">Start by generating your first cover letter!</div>
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}