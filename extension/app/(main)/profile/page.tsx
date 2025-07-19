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
        return <div className="p-8 text-center">Please sign in to view your profile.</div>;
    }

    return (
        <div className="max-w-2xl mx-auto py-12">
            <Card>
                <CardContent>
                    <div className="flex items-center gap-4 mb-6">
                        <Image src={user.image || "/default-avatar.png"} alt="Profile" width={64} height={64} className="rounded-full" />
                        <div>
                            <div className="text-xl font-bold">{user.name}</div>
                            <div className="text-gray-600">{user.email}</div>
                            <div className="text-sm text-[#419D78] mt-1">Plan: Pro</div>
                        </div>
                    </div>
                    {/* Show uploaded resume info */}
                    <ProfileUploads
                        userId={user.id || user.email || ''}
                        resumeInfo={profile.resumeInfo || undefined}
                        githubUsername={profile.githubUsername}
                        onRefresh={() => {
                            if (user.id) fetchUserProfile(user.id).then(data => setProfile(data as ProfileData));
                        }}
                    />
                    {/* Show submitted GitHub username */}
                    <div className="mt-6">
                        <div className="font-semibold mb-2">Submitted GitHub Username</div>
                        {profile.githubUsername ? (
                            <div className="bg-gray-100 rounded px-4 py-2 text-gray-800 inline-block">
                                {profile.githubUsername}
                            </div>
                        ) : (
                            <div className="text-gray-500">No GitHub username submitted yet.</div>
                        )}
                    </div>
                    {/* Recent Cover Letters */}
                    <div className="mt-6">
                        <div className="font-semibold mb-2">Recent Cover Letters</div>
                        {(Array.isArray(profile.recentCoverLetters) && profile.recentCoverLetters.length > 0) ? (
                            <ul className="list-disc pl-5">
                                {profile.recentCoverLetters.map((cl, idx) => (
                                    <li key={idx}>{cl.title} - {cl.date}</li>
                                ))}
                            </ul>
                        ) : (
                            <div className="text-gray-500">No cover letters generated yet.</div>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}