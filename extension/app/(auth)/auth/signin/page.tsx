'use client'

import { getProviders, signIn } from "next-auth/react"
import { useState, useEffect } from "react"
import type { ClientSafeProvider, LiteralUnion } from "next-auth/react"
import type { BuiltInProviderType } from "next-auth/providers/index"
import { LucideChrome, Github } from 'lucide-react'
import Image from "next/image"
import { Button } from "../../../components/ui/Button"
import { Card, CardContent } from "../../../components/ui/Card"

interface ProviderDetails {
  id: string;
  name: string;
  icon?: React.ComponentType<any>;
  bgColor?: string;
  textColor?: string;
}

const providerDetailsMap: Record<string, ProviderDetails> = {
  google: { 
    id: 'google', 
    name: 'Google',
    bgColor: 'bg-white dark:bg-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-gray-700', 
    textColor: 'text-gray-700 dark:text-gray-300',
    icon: LucideChrome
  },
  github: { 
    id: 'github', 
    name: 'GitHub', 
    bgColor: 'bg-gray-800 hover:bg-gray-900 dark:bg-gray-700 dark:hover:bg-gray-600', 
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
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-gray-900 dark:via-[#111111] dark:to-gray-800 flex justify-center items-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#419D78]"></div>
      </div>
    )
  }

  const oauthProviders = Object.values(providers).filter(
    p => p.type === 'oauth' && (p.id === 'google' || p.id === 'github')
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-gray-900 dark:via-[#111111] dark:to-gray-800 flex flex-col items-center justify-center p-4">
      {/* Background Elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-blue-200 dark:bg-blue-900/30 rounded-full opacity-20 animate-bounce"></div>
      <div className="absolute top-40 right-20 w-16 h-16 bg-green-200 dark:bg-green-900/30 rounded-full opacity-20 animate-bounce" style={{ animationDelay: '1s' }}></div>
      <div className="absolute bottom-20 left-20 w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full opacity-20 animate-bounce" style={{ animationDelay: '2s' }}></div>

      <Card className="w-full max-w-md animate-fadeIn bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800">
        <CardContent className="p-8">
          {/* Logo and Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <Image
                src="/Neoterik-Genesis.png"
                alt="Neoterik.ai Logo"
                width={60}
                height={60}
                priority
                className="rounded-xl"
              />
            </div>
            <h1 className="text-2xl font-bold text-[#2D3047] dark:text-gray-100 mb-2">Welcome to Neoterik.ai</h1>
            <p className="text-gray-600 dark:text-gray-400">Sign in to start creating amazing cover letters</p>
          </div>

          {/* Social Login Buttons */}
          <div className="space-y-3">
            {oauthProviders.map((provider) => {
              const details = providerDetailsMap[provider.id] || { 
                id: provider.id, 
                name: provider.name, 
                bgColor: 'bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700', 
                textColor: 'text-white' 
              };
              
              const IconComponent = details.icon;
              
              return (
                <button
                  key={details.id}
                  onClick={() => signIn(details.id, { callbackUrl: 'http://localhost:3000/auth/extension-callback/' })}
                  className={`w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg font-medium transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${details.bgColor} ${details.textColor}`}
                >
                  {IconComponent && <IconComponent className="w-5 h-5" />}
                  Continue with {details.name}
                </button>
              );
            })}
          </div>

          {/* Divider */}
          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200 dark:border-gray-700"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white dark:bg-[#1a1a1a] text-gray-500 dark:text-gray-400">or</span>
            </div>
          </div>

          {/* Email Form */}
          <form className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Email address
              </label>
              <input
                id="email"
                type="email"
                placeholder="Enter your email"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 bg-white dark:bg-[#2a2a2a] text-gray-900 dark:text-gray-100 rounded-lg focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none transition-colors"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Password
              </label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 bg-white dark:bg-[#2a2a2a] text-gray-900 dark:text-gray-100 rounded-lg focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none transition-colors"
              />
            </div>
            <Button type="submit" className="w-full">
              Sign In
            </Button>
          </form>

          {/* Footer Links */}
          <div className="mt-6 text-center space-y-2">
            <a href="#" className="text-sm text-[#419D78] hover:underline">
              Forgot your password?
            </a>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Don't have an account?{' '}
              <a href="#" className="text-[#419D78] hover:underline font-medium">
                Sign up
              </a>
            </div>
          </div>

          {/* Terms */}
          <div className="mt-6 text-xs text-gray-500 dark:text-gray-400 text-center">
            By signing in, you agree to our{' '}
            <a href="#" className="text-[#419D78] hover:underline">Terms of Service</a>
            {' '}and{' '}
            <a href="#" className="text-[#419D78] hover:underline">Privacy Policy</a>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <p className="mt-8 text-center text-xs text-gray-500 dark:text-gray-400">
        &copy; 2025 Neoterik.ai. All rights reserved.
      </p>
    </div>
  )
}