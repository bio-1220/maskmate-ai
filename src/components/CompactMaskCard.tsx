import React from 'react';
import { cn } from '@/lib/utils';
import { findMaskInfo } from '@/data/maskInfo';

interface CompactMaskCardProps {
  rank: number;
  maskName: string;
  maskImage: string;
  combinedScore: number;
  isExpanded: boolean;
  onClick: () => void;
}

// 탈 이름에서 숫자만 제거
const extractMaskName = (name: string): string => {
  return name.replace(/\d+/g, '').trim();
};

export const CompactMaskCard: React.FC<CompactMaskCardProps> = ({
  rank,
  maskName,
  maskImage,
  combinedScore,
  isExpanded,
  onClick,
}) => {
  const cleanMaskName = extractMaskName(maskName);
  const maskInfo = findMaskInfo(cleanMaskName);
  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <div
      onClick={onClick}
      className={cn(
        "relative bg-card rounded-xl overflow-hidden shadow-card cursor-pointer transition-all duration-300",
        "hover:shadow-glow hover:scale-[1.02]",
        isExpanded && "ring-2 ring-primary"
      )}
    >
      {/* Rank Badge */}
      <div className={cn(
        "absolute top-3 left-3 z-10 w-8 h-8 rounded-full flex items-center justify-center font-serif font-bold text-sm shadow-soft",
        "bg-gradient-to-br from-muted to-secondary text-foreground"
      )}>
        {rank}
      </div>

      <div className="flex items-center gap-4 p-4">
        {/* Small Mask Image */}
        <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-secondary to-muted overflow-hidden flex-shrink-0">
          <img 
            src={maskImage} 
            alt={maskName}
            className="w-full h-full object-contain p-1"
            onError={(e) => {
              (e.target as HTMLImageElement).src = '/placeholder.svg';
            }}
          />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <h4 className="font-serif text-lg font-semibold text-foreground truncate">
            {maskInfo?.koreanName || cleanMaskName}
          </h4>
          {maskInfo && (
            <p className="text-xs text-muted-foreground">{maskInfo.origin}</p>
          )}
        </div>

        {/* Score */}
        <div className="text-right flex-shrink-0">
          <div className="text-xl font-bold text-primary">
            {formatScore(combinedScore)}%
          </div>
          <p className="text-xs text-muted-foreground">점수</p>
        </div>
      </div>

      {/* Expand hint */}
      <div className="px-4 pb-3 text-center">
        <span className="text-xs text-muted-foreground">
          클릭하여 자세히 보기
        </span>
      </div>
    </div>
  );
};