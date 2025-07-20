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
    <div className="min-h-screen bg-[#111111] text-gray-300">
      {/* Hero Section */}
      <section className="relative text-center py-24 sm:py-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Subtle background glow for atmosphere */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[60%] bg-gradient-to-t from-[#419D78]/10 to-transparent blur-3xl -z-0"></div>
        
        <div className="max-w-4xl mx-auto relative z-10">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-50 tracking-tight">
            Career Insights &
            <span className="bg-gradient-to-r from-[#419D78] to-[#6ddaa8] bg-clip-text text-transparent block mt-2"> Expert Advice</span>
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-gray-400 leading-8">
            Stay ahead in your career with actionable insights, industry trends, 
            and expert tips from our team of career professionals.
          </p>
        </div>
      </section>

      {/* Featured Post */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <Card className="overflow-hidden bg-[#1a1a1a] border border-gray-800">
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
                  <span className="text-sm text-gray-400">{featuredPost.category}</span>
                </div>
                <h2 className="text-3xl font-bold text-gray-100 mb-4">
                  {featuredPost.title}
                </h2>
                <p className="text-gray-400 mb-6 leading-relaxed">
                  {featuredPost.excerpt}
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-700 rounded-full"></div>
                    <div>
                      <div className="font-medium text-gray-200">{featuredPost.author}</div>
                      <div className="text-sm text-gray-400">{featuredPost.date} • {featuredPost.readTime}</div>
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
                    : 'bg-[#1a1a1a] text-gray-400 hover:bg-gray-800 hover:text-gray-100 border border-gray-800'
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
              <Card key={index} className="blog-card bg-[#1a1a1a] border border-gray-800 hover:border-[#419D78] transition-colors duration-300">
                <div className="relative h-48">
                  <Image
                    src={post.image}
                    alt={post.title}
                    fill
                    className="blog-image object-cover rounded-t-lg"
                  />
                </div>
                <div className="blog-content p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xs font-medium text-[#419D78] bg-[#419D78]/20 px-2 py-1 rounded">
                      {post.category}
                    </span>
                  </div>
                  <h3 className="blog-title text-lg font-semibold mb-2 text-gray-100">{post.title}</h3>
                  <p className="blog-excerpt text-gray-400 text-sm mb-4 leading-relaxed">{post.excerpt}</p>
                  <div className="blog-meta flex items-center justify-between text-xs text-gray-500">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 bg-gray-700 rounded-full"></div>
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
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-100 mb-4">
            Stay Updated with Career Insights
          </h2>
          <p className="text-lg text-gray-400 mb-8">
            Get the latest career tips, industry trends, and job search strategies 
            delivered to your inbox every week.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-4 py-3 border border-gray-700 bg-[#1a1a1a] text-gray-300 rounded-lg focus:ring-2 focus:ring-[#419D78] focus:border-[#419D78] outline-none placeholder-gray-500"
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
      <footer className="border-t border-gray-800 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-500">
          <p>&copy; {new Date().getFullYear()} Neoterik.ai. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}