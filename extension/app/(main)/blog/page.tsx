'use client'

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '../../components/ui/Button';
import { Card, CardContent } from '../../components/ui/Card';

export default function BlogPage() {
  const featuredPost = {
    title: 'The Ultimate Guide to Writing Cover Letters in 2025',
    excerpt: 'Discover the latest trends and best practices for creating compelling cover letters that get you noticed by hiring managers.',
    image: 'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?w=800&h=400&fit=crop',
    author: 'Sarah Chen',
    date: 'January 15, 2025',
    readTime: '8 min read',
    category: 'Career Tips'
  };

  const blogPosts = [
    {
      title: 'How AI is Revolutionizing Job Applications',
      excerpt: 'Explore how artificial intelligence is transforming the way we apply for jobs and what it means for job seekers.',
      image: 'https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?w=400&h=250&fit=crop',
      author: 'Michael Rodriguez',
      date: 'January 12, 2025',
      readTime: '6 min read',
      category: 'Technology'
    },
    {
      title: '10 Common Cover Letter Mistakes to Avoid',
      excerpt: 'Learn about the most frequent errors job seekers make in their cover letters and how to avoid them.',
      image: 'https://images.pexels.com/photos/3184291/pexels-photo-3184291.jpeg?w=400&h=250&fit=crop',
      author: 'Emily Johnson',
      date: 'January 10, 2025',
      readTime: '5 min read',
      category: 'Career Tips'
    },
    {
      title: 'Tailoring Your Application for Remote Work',
      excerpt: 'Specific strategies for applying to remote positions and highlighting your remote work capabilities.',
      image: 'https://images.pexels.com/photos/4050315/pexels-photo-4050315.jpeg?w=400&h=250&fit=crop',
      author: 'David Kim',
      date: 'January 8, 2025',
      readTime: '7 min read',
      category: 'Remote Work'
    },
    {
      title: 'Industry-Specific Cover Letter Tips',
      excerpt: 'How to customize your cover letter for different industries, from tech to healthcare to finance.',
      image: 'https://images.pexels.com/photos/3184465/pexels-photo-3184465.jpeg?w=400&h=250&fit=crop',
      author: 'Lisa Wang',
      date: 'January 5, 2025',
      readTime: '9 min read',
      category: 'Industry Insights'
    },
    {
      title: 'The Psychology of Hiring Managers',
      excerpt: 'Understanding what hiring managers really look for in applications and how to appeal to their decision-making process.',
      image: 'https://images.pexels.com/photos/3184339/pexels-photo-3184339.jpeg?w=400&h=250&fit=crop',
      author: 'James Thompson',
      date: 'January 3, 2025',
      readTime: '6 min read',
      category: 'Psychology'
    },
    {
      title: 'Building Your Personal Brand Through Applications',
      excerpt: 'How to consistently communicate your personal brand across all job application materials.',
      image: 'https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg?w=400&h=250&fit=crop',
      author: 'Rachel Green',
      date: 'December 30, 2024',
      readTime: '8 min read',
      category: 'Personal Branding'
    }
  ];

  const categories = [
    'All Posts',
    'Career Tips',
    'Technology',
    'Remote Work',
    'Industry Insights',
    'Psychology',
    'Personal Branding'
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
              <Link href="/features" className="text-gray-600 hover:text-gray-900 transition-colors">Features</Link>
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
            Career Insights &
            <span className="bg-gradient-to-r from-[#419D78] to-[#E0A458] bg-clip-text text-transparent"> Expert Advice</span>
          </h1>
          <p className="text-xl text-gray-600 leading-relaxed">
            Stay ahead in your career with actionable insights, industry trends, 
            and expert tips from our team of career professionals.
          </p>
        </div>
      </section>

      {/* Featured Post */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <Card className="overflow-hidden">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-0">
              <div className="relative h-64 lg:h-auto">
                <Image
                  src={featuredPost.image}
                  alt={featuredPost.title}
                  fill
                  className="object-cover"
                />
              </div>
              <div className="p-8 lg:p-12">
                <div className="flex items-center gap-4 mb-4">
                  <span className="bg-[#419D78] text-white px-3 py-1 rounded-full text-sm font-medium">
                    Featured
                  </span>
                  <span className="text-sm text-gray-500">{featuredPost.category}</span>
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  {featuredPost.title}
                </h2>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  {featuredPost.excerpt}
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                    <div>
                      <div className="font-medium text-gray-900">{featuredPost.author}</div>
                      <div className="text-sm text-gray-500">{featuredPost.date} • {featuredPost.readTime}</div>
                    </div>
                  </div>
                  <Button>Read More</Button>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* Categories Filter */}
      <section className="py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-wrap gap-3 justify-center">
            {categories.map((category, index) => (
              <button
                key={index}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  index === 0
                    ? 'bg-[#419D78] text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Blog Posts Grid */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {blogPosts.map((post, index) => (
              <Card key={index} className="blog-card">
                <div className="relative h-48">
                  <Image
                    src={post.image}
                    alt={post.title}
                    fill
                    className="blog-image"
                  />
                </div>
                <div className="blog-content">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xs font-medium text-[#419D78] bg-green-50 px-2 py-1 rounded">
                      {post.category}
                    </span>
                  </div>
                  <h3 className="blog-title">{post.title}</h3>
                  <p className="blog-excerpt">{post.excerpt}</p>
                  <div className="blog-meta">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 bg-gray-200 rounded-full"></div>
                      <span>{post.author}</span>
                    </div>
                    <span>{post.date} • {post.readTime}</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Newsletter Signup */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Stay Updated with Career Insights
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            Get the latest career tips, industry trends, and job search strategies 
            delivered to your inbox every week.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none"
            />
            <Button className="whitespace-nowrap">
              Subscribe
            </Button>
          </div>
          <p className="text-sm text-gray-500 mt-4">
            No spam, unsubscribe at any time.
          </p>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Put These Tips into Action?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Use our AI-powered cover letter generator to create compelling applications 
            that incorporate these best practices.
          </p>
          <Link href="/auth/signin">
            <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
              Start Creating Cover Letters
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
                <li><Link href="/blog" className="hover:text-white transition-colors">Blog</Link></li>
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