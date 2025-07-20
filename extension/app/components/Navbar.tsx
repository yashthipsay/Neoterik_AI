"use client";

import Image from 'next/image';
import Link from 'next/link';
import { useSession, signOut } from "next-auth/react";
import { usePathname, useRouter } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';
import { Menu, X } from 'lucide-react';

// NOTE: You only need to install lucide-react for the icons
// npm install lucide-react

export function Navbar() {
    const { data: session, status } = useSession();
    const pathname = usePathname();
    const router = useRouter();

    const [isMobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [isProfileMenuOpen, setProfileMenuOpen] = useState(false);
    
    const profileMenuRef = useRef(null);

    // Close the profile dropdown when clicking outside of it
    useEffect(() => {
        function handleClickOutside(event) {
            if (profileMenuRef.current && !profileMenuRef.current.contains(event.target)) {
                setProfileMenuOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [profileMenuRef]);


    const navLinks = [
        { href: "/", label: "Home" },
        { href: "/features", label: "Features" },
        { href: "/about", label: "About" },
        { href: "/pricing", label: "Pricing" },
        { href: "/blog", label: "Blog" },
    ];

    return (
        <>
            <nav className="fixed top-0 left-0 right-0 z-50 bg-[#111111]/80 backdrop-blur-lg border-b border-gray-800/80 shadow-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">

                        {/* Logo */}
                        <div className="flex-shrink-0">
                            <Link href="/" className="flex items-center gap-3 group">
                                <Image 
                                    src="/Neoterik-Genesis.png"
                                    alt="Neoterik.ai Logo" 
                                    width={32} 
                                    height={32} 
                                    className="rounded-lg transition-transform duration-300 group-hover:scale-110" 
                                />
                                <span className="text-xl font-bold text-gray-100">Neoterik.ai</span>
                            </Link>
                        </div>

                        {/* Desktop Navigation Links - Centered */}
                        <div className="hidden md:flex items-center justify-center flex-1">
                            <div className="flex items-center space-x-8">
                                {navLinks.map((link) => (
                                    <Link
                                        key={link.href}
                                        href={link.href}
                                        className={`relative group px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                                            pathname === link.href
                                                ? 'text-gray-100'
                                                : 'text-gray-400 hover:text-gray-100'
                                        }`}
                                    >
                                        {link.label}
                                        {/* CSS-based underline animation */}
                                        <span className={`absolute bottom-0 left-0 w-full h-0.5 bg-[#419D78] transition-transform duration-300 ease-out transform ${
                                            pathname === link.href ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100 origin-center'
                                        }`} />
                                    </Link>
                                ))}
                            </div>
                        </div>

                        {/* Auth Buttons and Profile Dropdown */}
                        <div className="hidden md:flex items-center gap-4">
                            {status === "authenticated" && session?.user ? (
                                <div className="relative" ref={profileMenuRef}>
                                    <button onClick={() => setProfileMenuOpen(!isProfileMenuOpen)} className="flex items-center gap-2 rounded-full p-1 transition-colors hover:bg-gray-800">
                                        <Image
                                            src={session.user.image || "/default-avatar.png"}
                                            alt="Profile"
                                            width={32}
                                            height={32}
                                            className="rounded-full border-2 border-transparent group-hover:border-gray-700"
                                        />
                                    </button>
                                    {/* Profile dropdown with CSS transitions */}
                                    <div className={`absolute right-0 mt-2 w-48 bg-[#1a1a1a] rounded-md shadow-lg py-1 border border-gray-800/80 transition-all duration-200 ease-out transform ${
                                        isProfileMenuOpen ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'
                                    }`}>
                                        <div className="px-4 py-2 border-b border-gray-800">
                                            <p className="text-sm font-semibold text-gray-200 truncate">{session.user.name}</p>
                                            <p className="text-xs text-gray-400 truncate">{session.user.email}</p>
                                        </div>
                                        <Link href="/profile" className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-800" onClick={() => setProfileMenuOpen(false)}>Profile</Link>
                                        <button onClick={() => { signOut(); setProfileMenuOpen(false); }} className="w-full text-left block px-4 py-2 text-sm text-red-400 hover:bg-red-950/20">Sign Out</button>
                                    </div>
                                </div>
                            ) : (
                                <button onClick={() => router.push('/auth/signin')} className="px-4 py-2 text-sm font-medium text-white bg-[#419D78] rounded-md hover:bg-[#37876A] transition-all duration-300">
                                    Sign In
                                </button>
                            )}
                        </div>
                        
                        {/* Mobile Menu Button */}
                        <div className="md:hidden flex items-center">
                            <button onClick={() => setMobileMenuOpen(!isMobileMenuOpen)} className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-100 hover:bg-gray-800">
                                {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Menu with CSS transition */}
                <div className={`md:hidden overflow-hidden transition-all duration-300 ease-in-out ${isMobileMenuOpen ? 'max-h-96' : 'max-h-0'}`}>
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-800/80">
                        {navLinks.map((link) => (
                            <Link
                                key={link.href}
                                href={link.href}
                                onClick={() => setMobileMenuOpen(false)}
                                className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                                    pathname === link.href ? 'bg-gray-800 text-gray-100' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'
                                }`}
                            >
                                {link.label}
                            </Link>
                        ))}
                        <div className="pt-4 mt-4 border-t border-gray-800/80">
                            {status === "authenticated" && session?.user ? (
                                <div className="flex items-center px-3">
                                    <div className="flex-shrink-0">
                                        <Image src={session.user.image || "/default-avatar.png"} alt="Profile" width={40} height={40} className="rounded-full" />
                                    </div>
                                    <div className="ml-3">
                                        <div className="text-base font-medium text-gray-200">{session.user.name}</div>
                                        <div className="text-sm font-medium text-gray-400">{session.user.email}</div>
                                    </div>
                                </div>
                            ) : (
                                <button onClick={() => { router.push('/auth/signin'); setMobileMenuOpen(false); }} className="w-full text-left block px-3 py-2 rounded-md text-base font-medium text-gray-400 hover:bg-gray-800 hover:text-gray-100">
                                    Sign In
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </nav>
            {/* Add padding to the top of the page content to offset the fixed navbar */}
            <div className="h-16" /> 
        </>
    );
}