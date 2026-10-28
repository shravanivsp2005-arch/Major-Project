import React from 'react';
import { cn } from '@/lib/utils';

export const Logo = ({ className, size = 'md', showText = true, tone = 'default' }) => {
  const dims = { sm: 'h-12 w-12', md: 'h-16 w-16', lg: 'h-24 w-24' }[size];
  return (
    <div className={cn('flex items-center justify-center overflow-hidden rounded-full bg-white', dims, className)}>
      <img src="/d1965f52-95ea-4665-9268-b2e387ff8646.png" alt="Smart City by Civic" className="h-full w-full object-contain" />
    </div>
  );
};
