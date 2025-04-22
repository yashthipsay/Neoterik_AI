'use client'

import { getProviders, signIn } from "next-auth/react"
import { useState, useEffect } from "react"
import type { ClientSafeProvider, LiteralUnion } from "next-auth/react"
import type { BuiltInProviderType } from "next-auth/providers/index"
import { LucideChrome, LucideIcon } from 'lucide-react'
import Image from "next/image"
import { Github } from 'lucide-react'

// Define provider details (can be shared or defined here)
interface ProviderDetails {
  id: string;
  name: string;
  icon?: string | LucideIcon;  // Updated to use LucideIcon type
  bgColor?: string;
  textColor?: string;
}

// Create a monochrome Google icon component
// const GoogleIcon = () => (
//   <svg viewBox="0 0 24 24" className="w-5 h-5 mr-3">
//     <path
//       fill="currentColor"
//       d="M21.35 11.1h-9.17v2.73h6.51c-.33 3.81-3.5 5.44-6.5 5.44C8.36 19.27 5 16.25 5 12c0-4.1 3.2-7.27 7.2-7.27 3.09 0 4.9 1.97 4.9 1.97L19 4.72S16.56 2 12.1 2C6.42 2 2.03 6.8 2.03 12c0 5.05 4.13 10 10.22 10 5.35 0 9.25-3.67 9.25-9.09 0-1.15-.15-1.81-.15-1.81z"
//     />
//   </svg>
// )

const providerDetailsMap: Record<string, ProviderDetails> = {
  google: { 
    id: 'google', 
    name: 'Google',
    bgColor: 'bg-white', 
    textColor: 'text-gray-700',
    icon: LucideChrome
  },
  github: { 
    id: 'github', 
    name: 'GitHub', 
    bgColor: 'bg-gray-800', 
    textColor: 'text-white',
    icon: Github 
  },
};

export default function SignIn() {
  const [providers, setProviders] = useState<Record<LiteralUnion<BuiltInProviderType>, ClientSafeProvider> | null>(null)

  useEffect(() => {
    async function fetchProviders() {
      const res = await getProviders()
      setProviders(res)
    }
    fetchProviders()
  }, [])

  if (!providers) {
    return <div className="flex justify-center items-center h-screen">Loading...</div>
  }

  // Filter to only show desired OAuth providers (e.g., Google, GitHub)
  const oauthProviders = Object.values(providers).filter(
    p => p.type === 'oauth' && (p.id === 'google' || p.id === 'github' || p.id === 'twitter') // Add/remove IDs as needed
  );

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900 p-4">
       <div className="flex items-center mb-8">
          <Image
            src="/Neoterik-Genesis.png"
            alt="Neoterik.ai Logo"
            width={50}
            height={50}
            priority
            className="mr-3"
          />
          <h1 className="text-2xl font-semibold text-[#2D3047] dark:text-[#E5E7EB]">Sign In</h1>
        </div>

      <div className="w-full max-w-sm space-y-4">
        {oauthProviders.map((provider) => {
          const details = providerDetailsMap[provider.id] || { 
            id: provider.id, 
            name: provider.name, 
            bgColor: 'bg-blue-500', 
            textColor: 'text-white' 
          };
          return (
            <button
              key={details.id}
              onClick={() => signIn(details.id, { callbackUrl: 'http://localhost:3000/auth/extension-callback/' })}
              className={`w-full flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors ${details.bgColor} ${details.textColor} hover:opacity-90`}
            >
              {details.icon && (
                typeof details.icon === 'string' ? (
                  <Image src={details.icon} alt={`${details.name} logo`} width={20} height={20} className="mr-3" />
                ) : (
                  <details.icon />
                )
              )}
              Sign in with {details.name}
            </button>
          );
        })}
      </div>
       <p className="mt-6 text-center text-xs text-gray-500 dark:text-gray-400">
          &copy; 2025 Neoterik.ai
        </p>
    </div>
  )
}