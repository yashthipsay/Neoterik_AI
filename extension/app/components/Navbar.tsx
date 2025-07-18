"use client";
import Image from 'next/image';
import Link from 'next/link';
import { useSession, signOut } from "next-auth/react";
import { useRouter } from 'next/navigation';
import { Button } from './ui/Button';
import { StatusBadge } from './ui/StatusBadge';

export function Navbar() {
    const { data: session, status } = useSession();
    const router = useRouter();

    return (
        <nav className="flex items-center justify-between p-4 border-b border-gray-200 bg-white sticky top-0 z-10">
            <Link href="/" className="flex items-center gap-3">
                <Image src="/Neoterik-Genesis.png" alt="Neoterik.ai Logo" width={32} height={32} className="rounded-lg" />
                <span className="text-xl font-bold text-[#2D3047]">Neoterik.ai</span>
            </Link>
            <div className="flex items-center gap-6">
                <Link href="/" className="text-gray-600 hover:text-gray-900 transition-colors">Home</Link>
                <Link href="/features" className="text-gray-600 hover:text-gray-900 transition-colors">Features</Link>
                <Link href="/about" className="text-gray-600 hover:text-gray-900 transition-colors">About</Link>
                <Link href="/blog" className="text-gray-600 hover:text-gray-900 transition-colors">Blog</Link>
                <Link href="/contact" className="text-gray-600 hover:text-gray-900 transition-colors">Contact</Link>
                <Link href="/pricing" className="text-gray-600 hover:text-gray-900 transition-colors">Pricing</Link>
                {status === "authenticated" && session?.user ? (
                    <div className="flex items-center gap-2">
                        <button
                            className="flex items-center gap-2 cursor-pointer"
                            onClick={() => router.push('/profile')}
                            title="Profile"
                        >
                            <Image
                                src={session.user.image || "/default-avatar.png"}
                                alt="Profile"
                                width={28}
                                height={28}
                                className="rounded-full"
                            />
                            <span className="text-xs font-medium">{session.user.name?.split(' ')[0]}</span>
                            <StatusBadge status="success" className="text-xs">Pro</StatusBadge>
                        </button>
                        <Button size="sm" variant="outline" onClick={() => signOut()} className="text-xs">Sign out</Button>
                    </div>
                ) : (
                    <Button size="sm" onClick={() => router.push('/auth/signin')}>Sign in</Button>
                )}
            </div>
        </nav>
    );
}