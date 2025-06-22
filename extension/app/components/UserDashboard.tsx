'use client'

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { StatusBadge } from './ui/StatusBadge';
import { ProgressBar } from './ui/ProgressBar';

interface UserStats {
  coverLettersGenerated: number;
  monthlyLimit: number;
  successRate: number;
  averageRating: number;
}

interface UserDashboardProps {
  stats: UserStats;
  recentActivity: any[];
}

export const UserDashboard: React.FC<UserDashboardProps> = ({ stats, recentActivity }) => {
  const usagePercentage = (stats.coverLettersGenerated / stats.monthlyLimit) * 100;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="text-center" parametric={true} parametricIntensity="low">
          <div className="stat-number text-[#419D78]">{stats.coverLettersGenerated}</div>
          <div className="stat-label">Cover Letters Generated</div>
        </Card>
        
        <Card className="text-center" parametric={true} parametricIntensity="low">
          <div className="stat-number text-[#E0A458]">{stats.successRate}%</div>
          <div className="stat-label">Success Rate</div>
        </Card>
        
        <Card className="text-center" parametric={true} parametricIntensity="low">
          <div className="stat-number text-[#419D78]">{stats.averageRating}/5</div>
          <div className="stat-label">Average Rating</div>
        </Card>
        
        <Card className="text-center" parametric={true} parametricIntensity="low">
          <div className="stat-number text-[#E0A458]">{stats.monthlyLimit - stats.coverLettersGenerated}</div>
          <div className="stat-label">Remaining This Month</div>
        </Card>
      </div>

      {/* Usage Progress */}
      <Card parametric={true} parametricIntensity="medium">
        <CardHeader parametric={true}>
          <CardTitle parametric={true}>Monthly Usage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Used: {stats.coverLettersGenerated} / {stats.monthlyLimit}</span>
              <span className="font-semibold text-[#419D78]">{Math.round(usagePercentage)}%</span>
            </div>
            <ProgressBar 
              value={usagePercentage} 
              color={usagePercentage > 80 ? 'warning' : 'primary'} 
            />
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card parametric={true} parametricIntensity="high">
        <CardHeader parametric={true}>
          <CardTitle parametric={true}>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentActivity.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-[#419D78]/5 rounded-lg border border-[#419D78]/10">
                <div>
                  <div className="font-medium text-[#2D3047]">{activity.title}</div>
                  <div className="text-sm text-gray-600">{activity.company}</div>
                </div>
                <div className="text-right">
                  <StatusBadge status={activity.status}>{activity.statusText}</StatusBadge>
                  <div className="text-xs text-gray-500 mt-1">{activity.date}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};