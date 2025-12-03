import React from 'react';
import { Loader2 } from 'lucide-react';

export const LoadingState: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-fade-up">
      <div className="relative">
        <div className="w-20 h-20 rounded-full border-4 border-secondary" />
        <div className="absolute inset-0 w-20 h-20 rounded-full border-4 border-primary border-t-transparent animate-spin" />
        <div className="absolute inset-2 w-16 h-16 rounded-full border-4 border-gold/30 border-b-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
      </div>
      <p className="mt-6 text-lg font-medium text-foreground">
        어울리는 탈을 찾고 있습니다...
      </p>
      <p className="mt-2 text-sm text-muted-foreground">
        AI가 당신의 얼굴과 표정을 분석하고 있어요
      </p>
    </div>
  );
};
