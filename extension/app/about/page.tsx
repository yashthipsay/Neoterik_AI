'use client'

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';

export default function AboutPage() {
  const team = [
    {
      name: 'Alex Chen',
      role: 'CEO & Co-Founder',
      image: 'https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?w=300&h=300&fit=crop&crop=face',
      bio: 'Former Google engineer with 10+ years in AI and machine learning. Passionate about democratizing career opportunities.'
    },
    {
      name: 'Sarah Martinez',
      role: 'CTO & Co-Founder',
      image: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?w=300&h=300&fit=crop&crop=face',
      bio: 'AI researcher and former Microsoft principal engineer. Expert in natural language processing and career technology.'
    },
    {
      name: 'David Kim',
      role: 'Head of Product',
      image: 'https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg?w=300&h=300&fit=crop&crop=face',
      bio: 'Product leader with experience at LinkedIn and Uber. Focused on creating intuitive user experiences.'
    },
    {
      name: 'Emily Rodriguez',
      role: 'Head of Design',
      image: 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?w=300&h=300&fit=crop&crop=face',
      bio: 'Design expert with a background in user experience and human-computer interaction. Believes in design that empowers.'
    }
  ];

  const values = [
    {
      icon: '🎯',
      title: 'Mission-Driven',
      description: 'We believe everyone deserves equal access to career opportunities, regardless of their background or writing skills.'
    },
    {
      icon: '🤝',
      title: 'User-Centric',
      description: 'Every feature we build is designed with our users\' success in mind. Your career growth is our primary goal.'
    },
    {
      icon: '🔬',
      title: 'Innovation First',
      description: 'We leverage cutting-edge AI technology to solve real-world career challenges and stay ahead of industry trends.'
    },
    {
      icon: '🌍',
      title: 'Global Impact',
      description: 'Our tools are designed to help job seekers worldwide, breaking down barriers and creating opportunities.'
    }
  ];

  const milestones = [
    {
      year: '2023',
      title: 'Company Founded',
      description: 'Neoterik.ai was founded with the vision of democratizing career opportunities through AI.'
    },
    {
      year: '2024',
      title: 'Product Launch',
      description: 'Launched our first AI-powered cover letter generator, helping thousands of job seekers.'
    },
    {
      year: '2024',
      title: '10K Users',
      description: 'Reached 10,000 active users and generated over 50,000 cover letters.'
    },
    {
      year: '2025',
      title: 'Series A Funding',
      description: 'Raised $5M in Series A funding to expand our AI capabilities and team.'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
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
              <Link href="/features" className="text-gray-600 hover:text-gray-900 transition-colors">Features</Link>
              <Link href="/pricing" className="text-gray-600 hover:text-gray-900 transition-colors">Pricing</Link>
              <Link href="/auth/signin">
                <Button>Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            About Neoterik.ai
          </h1>
          <p className="text-xl text-gray-600 leading-relaxed">
            We're on a mission to democratize career opportunities by making professional 
            cover letter writing accessible to everyone through the power of artificial intelligence.
          </p>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 mb-6">Our Mission</h2>
              <p className="text-lg text-gray-600 mb-6">
                At Neoterik.ai, we believe that everyone deserves equal access to career opportunities. 
                Too often, talented individuals are overlooked simply because they struggle with 
                articulating their value in writing.
              </p>
              <p className="text-lg text-gray-600 mb-6">
                Our AI-powered platform levels the playing field by helping job seekers create 
                compelling, personalized cover letters that showcase their unique strengths and 
                align with employer expectations.
              </p>
              <div className="flex gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#419D78]">50K+</div>
                  <div className="text-sm text-gray-600">Cover Letters</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#419D78]">10K+</div>
                  <div className="text-sm text-gray-600">Happy Users</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#419D78]">85%</div>
                  <div className="text-sm text-gray-600">Success Rate</div>
                </div>
              </div>
            </div>
            <div className="relative">
              <Image
                src="https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?w=600&h=400&fit=crop"
                alt="Team collaboration"
                width={600}
                height={400}
                className="rounded-2xl shadow-2xl"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Values</h2>
            <p className="text-lg text-gray-600">
              The principles that guide everything we do
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {values.map((value, index) => (
              <Card key={index} className="p-8">
                <CardContent>
                  <div className="text-4xl mb-4">{value.icon}</div>
                  <h3 className="text-xl font-semibold mb-3">{value.title}</h3>
                  <p className="text-gray-600">{value.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Meet Our Team</h2>
            <p className="text-lg text-gray-600">
              The passionate individuals behind Neoterik.ai
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {team.map((member, index) => (
              <Card key={index} className="text-center p-6">
                <CardContent>
                  <Image
                    src={member.image}
                    alt={member.name}
                    width={120}
                    height={120}
                    className="rounded-full mx-auto mb-4"
                  />
                  <h3 className="text-lg font-semibold mb-1">{member.name}</h3>
                  <p className="text-[#419D78] font-medium mb-3">{member.role}</p>
                  <p className="text-sm text-gray-600">{member.bio}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Journey</h2>
            <p className="text-lg text-gray-600">
              Key milestones in our mission to transform career opportunities
            </p>
          </div>
          
          <div className="space-y-8">
            {milestones.map((milestone, index) => (
              <div key={index} className="flex items-start gap-6">
                <div className="flex-shrink-0 w-20 h-20 bg-[#419D78] text-white rounded-full flex items-center justify-center font-bold">
                  {milestone.year}
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold mb-2">{milestone.title}</h3>
                  <p className="text-gray-600">{milestone.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Join Our Mission
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Ready to transform your career with AI-powered tools?
          </p>
          <Link href="/auth/signin">
            <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
              Get Started Today
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
                <li><Link href="#" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Pricing</Link></li>
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