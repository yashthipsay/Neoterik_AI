'use client'

import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '../../components/ui/Button';
import { PricingCard } from '../../components/PricingCard';
import { Card, CardContent } from '../../components/ui/Card';

export default function PricingPage() {
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'yearly'>('monthly');

  const pricingTiers = [
    {
      name: 'Starter',
      price: billingCycle === 'monthly' ? '$9' : '$90',
      period: billingCycle === 'monthly' ? 'month' : 'year',
      description: 'Perfect for job seekers just getting started',
      features: [
        '10 cover letters per month',
        'Basic AI personalization',
        'Standard templates',
        'Email support',
        'PDF export'
      ],
      buttonText: 'Start Free Trial',
      buttonVariant: 'outline' as const
    },
    {
      name: 'Professional',
      price: billingCycle === 'monthly' ? '$19' : '$190',
      period: billingCycle === 'monthly' ? 'month' : 'year',
      description: 'Ideal for active job seekers and career changers',
      features: [
        '50 cover letters per month',
        'Advanced AI personalization',
        'Premium templates',
        'Priority support',
        'Multiple export formats',
        'Success analytics',
        'LinkedIn integration'
      ],
      popular: true,
      buttonText: 'Start Free Trial',
      buttonVariant: 'primary' as const
    },
    {
      name: 'Enterprise',
      price: billingCycle === 'monthly' ? '$49' : '$490',
      period: billingCycle === 'monthly' ? 'month' : 'year',
      description: 'For teams and career coaches',
      features: [
        'Unlimited cover letters',
        'Custom AI training',
        'White-label solution',
        'Dedicated support',
        'API access',
        'Team management',
        'Advanced analytics',
        'Custom integrations'
      ],
      buttonText: 'Contact Sales',
      buttonVariant: 'secondary' as const
    }
  ];

  const faqs = [
    {
      question: 'How does the free trial work?',
      answer: 'You get full access to all features for 7 days, no credit card required. After the trial, you can choose to upgrade or continue with our free tier.'
    },
    {
      question: 'Can I cancel my subscription anytime?',
      answer: 'Yes, you can cancel your subscription at any time. You\'ll continue to have access until the end of your billing period.'
    },
    {
      question: 'What makes your AI different?',
      answer: 'Our AI is specifically trained on successful cover letters and job descriptions. It understands industry nuances and creates highly personalized content.'
    },
    {
      question: 'Do you offer refunds?',
      answer: 'We offer a 30-day money-back guarantee. If you\'re not satisfied, we\'ll refund your payment, no questions asked.'
    },
    {
      question: 'Is my data secure?',
      answer: 'Absolutely. We use bank-level encryption and are SOC 2 Type II compliant. Your data is never shared with third parties.'
    },
    {
      question: 'Can I upgrade or downgrade my plan?',
      answer: 'Yes, you can change your plan at any time. Changes take effect immediately, and we\'ll prorate the billing accordingly.'
    }
  ];

  const handlePlanSelect = (tier: any) => {
    console.log('Selected plan:', tier.name);
    // Handle plan selection logic here
  };

  return (
    <div className="min-h-screen bg-[#111111] text-gray-300">
      {/* Hero Section */}
      <section className="relative text-center py-24 sm:py-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Subtle background glow for atmosphere */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[60%] bg-gradient-to-t from-[#419D78]/10 to-transparent blur-3xl -z-0"></div>
        
        <div className="max-w-4xl mx-auto relative z-10">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-50 tracking-tight">
            Simple, Transparent
            <span className="bg-gradient-to-r from-[#419D78] to-[#6ddaa8] bg-clip-text text-transparent block mt-2"> Pricing</span>
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-gray-400 leading-8 mb-8">
            Choose the perfect plan for your career goals. All plans include our core AI features 
            and come with a 7-day free trial.
          </p>
          
          {/* Billing Toggle */}
          <div className="flex items-center justify-center gap-4 mb-12">
            <span className={`font-medium ${billingCycle === 'monthly' ? 'text-gray-100' : 'text-gray-500'}`}>
              Monthly
            </span>
            <button
              onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'yearly' : 'monthly')}
              className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-[#419D78] focus:ring-offset-2"
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  billingCycle === 'yearly' ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
            <span className={`font-medium ${billingCycle === 'yearly' ? 'text-gray-100' : 'text-gray-500'}`}>
              Yearly
            </span>
            {billingCycle === 'yearly' && (
              <span className="bg-[#419D78] text-white text-xs font-medium px-2.5 py-0.5 rounded-full">
                Save 17%
              </span>
            )}
          </div>
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {pricingTiers.map((tier, index) => (
              <PricingCard
                key={index}
                tier={tier}
                onSelect={handlePlanSelect}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Features Comparison */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-100 mb-4">Compare All Features</h2>
            <p className="text-lg text-gray-400">
              See exactly what's included in each plan
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b-2 border-gray-700">
                  <th className="text-left py-4 px-6 font-semibold text-gray-100">Features</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-100">Starter</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-100">Professional</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-100">Enterprise</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">Cover letters per month</td>
                  <td className="py-4 px-6 text-center text-gray-400">10</td>
                  <td className="py-4 px-6 text-center text-gray-400">50</td>
                  <td className="py-4 px-6 text-center text-gray-400">Unlimited</td>
                </tr>
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">AI personalization</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                </tr>
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">Premium templates</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                </tr>
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">Success analytics</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                </tr>
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">API access</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                </tr>
                <tr>
                  <td className="py-4 px-6 font-medium text-gray-300">Team management</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-gray-600">-</td>
                  <td className="py-4 px-6 text-center text-[#419D78]">✓</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-100 mb-4">Frequently Asked Questions</h2>
            <p className="text-lg text-gray-400">
              Everything you need to know about our pricing and features
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
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-2xl font-bold text-gray-100 mb-4">Trusted by Professionals Worldwide</h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 items-center justify-items-center">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-300">SOC 2</div>
              <div className="text-sm text-gray-500">Compliant</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-300">GDPR</div>
              <div className="text-sm text-gray-500">Compliant</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-300">256-bit</div>
              <div className="text-sm text-gray-500">Encryption</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-300">99.9%</div>
              <div className="text-sm text-gray-500">Uptime</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Accelerate Your Career?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Start your free trial today and experience the power of AI-driven cover letters
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/auth/signin">
              <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
                Start Free Trial
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="text-lg px-8 py-4 border-white text-white hover:bg-white hover:text-[#419D78]">
              Contact Sales
            </Button>
          </div>
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