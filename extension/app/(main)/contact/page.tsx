'use client'

import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Textarea } from '../../components/ui/Textarea';
import { Card, CardContent } from '../../components/ui/Card';

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    setTimeout(() => {
      setIsSubmitting(false);
      alert('Thank you for your message! We\'ll get back to you soon.');
      setFormData({ name: '', email: '', subject: '', message: '' });
    }, 2000);
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const contactInfo = [
    {
      icon: '📧',
      title: 'Email Us',
      description: 'Send us an email and we\'ll respond within 24 hours',
      contact: 'hello@neoterik.ai'
    },
    {
      icon: '💬',
      title: 'Live Chat',
      description: 'Chat with our support team in real-time',
      contact: 'Available 9 AM - 6 PM PST'
    },
    {
      icon: '📞',
      title: 'Call Us',
      description: 'Speak directly with our team',
      contact: '+1 (555) 123-4567'
    },
    {
      icon: '📍',
      title: 'Visit Us',
      description: 'Come visit our headquarters',
      contact: 'San Francisco, CA'
    }
  ];

  const faqs = [
    {
      question: 'How quickly can I get started?',
      answer: 'You can start using Neoterik.ai immediately after signing up. Our onboarding process takes less than 5 minutes.'
    },
    {
      question: 'Do you offer enterprise solutions?',
      answer: 'Yes, we offer custom enterprise solutions for teams and organizations. Contact our sales team for more information.'
    },
    {
      question: 'Is there a mobile app?',
      answer: 'Currently, Neoterik.ai is available as a web application and browser extension. A mobile app is in development.'
    },
    {
      question: 'Can I integrate with my existing tools?',
      answer: 'We offer integrations with popular platforms like LinkedIn, Google Drive, and more. Check our integrations page for the full list.'
    }
  ];

  return (
    <div className="min-h-screen bg-[#111111] text-gray-300">
      {/* Hero Section */}
      <section className="relative text-center py-24 sm:py-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Subtle background glow for atmosphere */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[60%] bg-gradient-to-t from-[#419D78]/10 to-transparent blur-3xl -z-0"></div>
        
        <div className="max-w-4xl mx-auto relative z-10">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-50 tracking-tight">
            Get in
            <span className="bg-gradient-to-r from-[#419D78] to-[#6ddaa8] bg-clip-text text-transparent block mt-2"> Touch</span>
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-gray-400 leading-8">
            Have questions about Neoterik.ai? We're here to help. Reach out to our team 
            and we'll get back to you as soon as possible.
          </p>
        </div>
      </section>

      {/* Contact Info Cards */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {contactInfo.map((info, index) => (
              <Card key={index} className="text-center p-6 bg-[#1a1a1a] border border-gray-800 hover:border-[#419D78] transition-colors duration-300">
                <CardContent>
                  <div className="text-4xl mb-4">{info.icon}</div>
                  <h3 className="text-lg font-semibold mb-2 text-gray-100">{info.title}</h3>
                  <p className="text-gray-400 text-sm mb-3">{info.description}</p>
                  <p className="font-medium text-[#419D78]">{info.contact}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Form */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Form */}
            <Card className="p-8 bg-[#1a1a1a] border border-gray-800">
              <CardContent>
                <h2 className="text-2xl font-bold text-gray-100 mb-6">Send us a message</h2>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Name"
                      placeholder="Your full name"
                      value={formData.name}
                      onChange={(e) => handleInputChange('name', e.target.value)}
                      required
                    />
                    <Input
                      label="Email"
                      type="email"
                      placeholder="your@email.com"
                      value={formData.email}
                      onChange={(e) => handleInputChange('email', e.target.value)}
                      required
                    />
                  </div>
                  
                  <Input
                    label="Subject"
                    placeholder="What's this about?"
                    value={formData.subject}
                    onChange={(e) => handleInputChange('subject', e.target.value)}
                    required
                  />
                  
                  <Textarea
                    label="Message"
                    placeholder="Tell us more about your question or feedback..."
                    value={formData.message}
                    onChange={(e) => handleInputChange('message', e.target.value)}
                    rows={6}
                    required
                  />
                  
                  <Button
                    type="submit"
                    loading={isSubmitting}
                    disabled={isSubmitting}
                    className="w-full"
                  >
                    {isSubmitting ? 'Sending...' : 'Send Message'}
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Additional Info */}
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold text-gray-100 mb-4">Let's start a conversation</h2>
                <p className="text-gray-400 mb-6">
                  We're always excited to hear from our users and potential customers. 
                  Whether you have questions about our features, need technical support, 
                  or want to explore partnership opportunities, we're here to help.
                </p>
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-[#419D78]/20 rounded-full flex items-center justify-center">
                      <span className="text-[#419D78] text-sm">✓</span>
                    </div>
                    <span className="text-gray-300">Response within 24 hours</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-[#419D78]/20 rounded-full flex items-center justify-center">
                      <span className="text-[#419D78] text-sm">✓</span>
                    </div>
                    <span className="text-gray-300">Dedicated support team</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-[#419D78]/20 rounded-full flex items-center justify-center">
                      <span className="text-[#419D78] text-sm">✓</span>
                    </div>
                    <span className="text-gray-300">Personalized assistance</span>
                  </div>
                </div>
              </div>

              {/* Office Hours */}
              <Card className="p-6 bg-[#1a1a1a] border border-gray-800">
                <CardContent>
                  <h3 className="text-lg font-semibold mb-4 text-gray-100">Office Hours</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Monday - Friday</span>
                      <span className="font-medium text-gray-300">9:00 AM - 6:00 PM PST</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Saturday</span>
                      <span className="font-medium text-gray-300">10:00 AM - 4:00 PM PST</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sunday</span>
                      <span className="font-medium text-gray-300">Closed</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-100 mb-4">Quick Answers</h2>
            <p className="text-lg text-gray-400">
              Find answers to commonly asked questions
            </p>
          </div>
          
          <div className="space-y-6">
            {faqs.map((faq, index) => (
              <Card key={index} className="p-6 bg-[#1a1a1a] border border-gray-800 hover:border-[#419D78] transition-colors duration-300">
                <CardContent>
                  <h3 className="text-lg font-semibold mb-3 text-gray-100">{faq.question}</h3>
                  <p className="text-gray-400">{faq.answer}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="text-center mt-12">
            <p className="text-gray-400 mb-4">Can't find what you're looking for?</p>
            <Link href="/help">
              <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white">
                Visit Help Center
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Transform Your Job Search?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Don't wait - start creating compelling cover letters today
          </p>
          <Link href="/auth/signin">
            <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
              Get Started Now
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