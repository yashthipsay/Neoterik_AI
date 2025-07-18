'use client'

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';

export default function FeaturesPage() {
  const features = [
    {
      icon: '🤖',
      title: 'Advanced AI Technology',
      description: 'Our proprietary AI model is trained on thousands of successful cover letters and job descriptions.',
      details: [
        'Natural language processing for job description analysis',
        'Personalization based on your resume and experience',
        'Industry-specific language and terminology',
        'Continuous learning from successful applications'
      ]
    },
    {
      icon: '⚡',
      title: 'Lightning-Fast Generation',
      description: 'Generate professional cover letters in under 30 seconds with our optimized AI engine.',
      details: [
        'Sub-30 second generation time',
        'Real-time preview and editing',
        'Instant formatting and styling',
        'Quick iteration and refinement'
      ]
    },
    {
      icon: '🎯',
      title: 'Precision Targeting',
      description: 'Each cover letter is tailored to match specific job requirements and company culture.',
      details: [
        'Job description keyword matching',
        'Company culture analysis',
        'Role-specific skill highlighting',
        'Industry trend incorporation'
      ]
    },
    {
      icon: '📊',
      title: 'Success Analytics',
      description: 'Track your application success rate and optimize your job search strategy.',
      details: [
        'Application response rate tracking',
        'Interview conversion metrics',
        'A/B testing for different approaches',
        'Performance insights and recommendations'
      ]
    },
    {
      icon: '🔒',
      title: 'Enterprise Security',
      description: 'Bank-level security ensures your personal information is always protected.',
      details: [
        'End-to-end encryption',
        'SOC 2 Type II compliance',
        'GDPR and CCPA compliant',
        'Zero data retention policy'
      ]
    },
    {
      icon: '🌐',
      title: 'Multi-Language Support',
      description: 'Generate cover letters in multiple languages for global job opportunities.',
      details: [
        'Support for 15+ languages',
        'Cultural adaptation for different regions',
        'Local business etiquette integration',
        'Native speaker review available'
      ]
    }
  ];

  const integrations = [
    {
      name: 'LinkedIn',
      logo: 'https://images.pexels.com/photos/267350/pexels-photo-267350.jpeg?w=100&h=100&fit=crop',
      description: 'Import your LinkedIn profile data automatically'
    },
    {
      name: 'Indeed',
      logo: 'https://images.pexels.com/photos/267350/pexels-photo-267350.jpeg?w=100&h=100&fit=crop',
      description: 'Apply directly to Indeed job postings'
    },
    {
      name: 'Glassdoor',
      logo: 'https://images.pexels.com/photos/267350/pexels-photo-267350.jpeg?w=100&h=100&fit=crop',
      description: 'Access company insights and reviews'
    },
    {
      name: 'Google Drive',
      logo: 'https://images.pexels.com/photos/267350/pexels-photo-267350.jpeg?w=100&h=100&fit=crop',
      description: 'Save and organize your cover letters'
    }
  ];

  const useCases = [
    {
      title: 'Recent Graduates',
      description: 'Perfect for new graduates who need help articulating their potential and academic achievements.',
      icon: '🎓'
    },
    {
      title: 'Career Changers',
      description: 'Ideal for professionals transitioning to new industries or roles.',
      icon: '🔄'
    },
    {
      title: 'Senior Executives',
      description: 'Sophisticated language and executive-level positioning for leadership roles.',
      icon: '👔'
    },
    {
      title: 'International Candidates',
      description: 'Navigate cultural differences and language barriers with confidence.',
      icon: '🌍'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Navigation */}
      {/* <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center gap-3">
              <Image
                src="/Neoterik-Genesis.png"
                alt="Neoterik.ai Logo"
                width={40}
                height={40}
                className="rounded-lg"
              />
              <span className="text-xl font-bold text-gray-900">Neoterik.ai</span>
            </Link>
            <div className="hidden md:flex items-center gap-8">
              <Link href="/" className="text-gray-600 hover:text-gray-900 transition-colors">Home</Link>
              <Link href="/about" className="text-gray-600 hover:text-gray-900 transition-colors">About</Link>
              <Link href="/pricing" className="text-gray-600 hover:text-gray-900 transition-colors">Pricing</Link>
              <Link href="/auth/signin">
                <Button>Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav> */}

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Powerful Features for
            <span className="bg-gradient-to-r from-[#419D78] to-[#E0A458] bg-clip-text text-transparent"> Career Success</span>
          </h1>
          <p className="text-xl text-gray-600 leading-relaxed">
            Discover how our advanced AI technology and comprehensive feature set 
            can transform your job search and accelerate your career growth.
          </p>
        </div>
      </section>

      {/* Main Features */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {features.map((feature, index) => (
              <Card key={index} className="p-8 animate-fadeIn" style={{ animationDelay: `${index * 0.1}s` }}>
                <CardContent>
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{feature.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-2xl font-semibold mb-3">{feature.title}</h3>
                      <p className="text-gray-600 mb-4">{feature.description}</p>
                      <ul className="space-y-2">
                        {feature.details.map((detail, detailIndex) => (
                          <li key={detailIndex} className="flex items-center gap-2 text-sm text-gray-700">
                            <svg className="w-4 h-4 text-green-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                            {detail}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Perfect for Every Career Stage</h2>
            <p className="text-lg text-gray-600">
              Whether you're just starting out or leading a team, our AI adapts to your unique situation
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {useCases.map((useCase, index) => (
              <Card key={index} className="p-6">
                <CardContent>
                  <div className="flex items-start gap-4">
                    <div className="text-3xl">{useCase.icon}</div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">{useCase.title}</h3>
                      <p className="text-gray-600">{useCase.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Integrations */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Seamless Integrations</h2>
            <p className="text-lg text-gray-600">
              Connect with your favorite job search platforms and productivity tools
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {integrations.map((integration, index) => (
              <Card key={index} className="p-6 text-center">
                <CardContent>
                  <Image
                    src={integration.logo}
                    alt={integration.name}
                    width={60}
                    height={60}
                    className="rounded-lg mx-auto mb-4"
                  />
                  <h3 className="font-semibold mb-2">{integration.name}</h3>
                  <p className="text-sm text-gray-600">{integration.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            See It in Action
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            Watch how our AI transforms a simple job description into a compelling, 
            personalized cover letter in seconds.
          </p>
          <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
            <div className="aspect-video bg-gradient-to-br from-[#419D78] to-[#37876A] rounded-lg flex items-center justify-center">
              <div className="text-white text-center">
                <div className="text-6xl mb-4">▶️</div>
                <p className="text-xl font-semibold">Interactive Demo</p>
                <p className="opacity-90">Click to see Neoterik.ai in action</p>
              </div>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg">
              Try Interactive Demo
            </Button>
            <Link href="/auth/signin">
              <Button variant="outline" size="lg">
                Start Free Trial
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Experience These Features?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of professionals who have already transformed their job search with Neoterik.ai
          </p>
          <Link href="/auth/signin">
            <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
              Start Your Free Trial
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <Image
                  src="/Neoterik-Genesis.png"
                  alt="Neoterik.ai Logo"
                  width={32}
                  height={32}
                  className="rounded-lg"
                />
                <span className="text-xl font-bold">Neoterik.ai</span>
              </div>
              <p className="text-gray-400">
                AI-powered career tools to help you land your dream job.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/features" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">API</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/about" className="hover:text-white transition-colors">About</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Blog</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Careers</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="#" className="hover:text-white transition-colors">Help Center</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Contact</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Privacy</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Neoterik.ai. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}