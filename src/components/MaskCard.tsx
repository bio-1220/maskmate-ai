import React, { useState } from 'react';
import { Check, X, Sparkles, ChevronDown, ExternalLink, BookOpen } from 'lucide-react';
import { cn } from '@/lib/utils';
import { findMaskInfo, MaskDetailInfo } from '@/data/maskInfo';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';

interface MaskCardProps {
  rank: number;
  maskName: string;
  maskImage: string;
  cosineSimilarity: number;
  expressionMatch: boolean;
  combinedScore: number;
  faceExpression: string;
  maskExpression: string;
}

export const MaskCard: React.FC<MaskCardProps> = ({
  rank,
  maskName,
  maskImage,
  cosineSimilarity,
  expressionMatch,
  combinedScore,
  faceExpression,
  maskExpression,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const maskInfo = findMaskInfo(maskName);

  const getRankStyle = (rank: number) => {
    switch (rank) {
      case 1:
        return "from-gold to-dancheong-yellow text-warm-brown";
      case 2:
        return "from-muted to-secondary text-foreground";
      case 3:
        return "from-primary/30 to-primary/20 text-primary";
      default:
        return "from-muted to-secondary text-foreground";
    }
  };

  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <div 
      className={cn(
        "relative bg-card rounded-2xl overflow-hidden shadow-card transition-all duration-500 hover:shadow-glow",
        "animate-fade-up"
      )}
      style={{ animationDelay: `${rank * 150}ms` }}
    >
      {/* Rank Badge */}
      <div className={cn(
        "absolute top-4 left-4 z-10 w-10 h-10 rounded-full flex items-center justify-center font-serif font-bold text-lg shadow-soft",
        "bg-gradient-to-br",
        getRankStyle(rank)
      )}>
        {rank}
      </div>

      {/* Mask Image */}
      <div className="relative aspect-square bg-gradient-to-br from-secondary to-muted">
        <img 
          src={maskImage} 
          alt={maskName}
          className="w-full h-full object-contain p-4"
          onError={(e) => {
            (e.target as HTMLImageElement).src = '/placeholder.svg';
          }}
        />
        {rank === 1 && (
          <div className="absolute top-4 right-4">
            <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-gold/90 text-warm-brown text-xs font-medium">
              <Sparkles className="w-3 h-3" />
              최고 매칭
            </div>
          </div>
        )}
      </div>

      {/* Info Section */}
      <div className="p-5 space-y-4">
        {/* Mask Name */}
        <div className="flex items-center justify-between">
          <h3 className="font-serif text-xl font-semibold text-foreground">
            {maskInfo?.koreanName || maskName}
          </h3>
          {maskInfo && (
            <span className="text-xs text-muted-foreground px-2 py-1 bg-secondary rounded-full">
              {maskInfo.origin}
            </span>
          )}
        </div>

        {/* Scores */}
        <div className="space-y-3">
          {/* Combined Score */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">종합 점수</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-2 rounded-full bg-secondary overflow-hidden">
                <div 
                  className="h-full rounded-full bg-gradient-to-r from-primary to-gold transition-all duration-1000"
                  style={{ width: `${combinedScore * 100}%` }}
                />
              </div>
              <span className="text-lg font-bold text-primary">
                {formatScore(combinedScore)}%
              </span>
            </div>
          </div>

          {/* Cosine Similarity */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">시각적 유사도</span>
            <span className="text-sm font-medium text-foreground">
              {formatScore(cosineSimilarity)}%
            </span>
          </div>

          {/* Expression Match */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">표정 일치</span>
            <div className="flex items-center gap-2">
              {expressionMatch ? (
                <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-dancheong-green/20 text-dancheong-green">
                  <Check className="w-3 h-3" />
                  <span className="text-xs font-medium">일치</span>
                </div>
              ) : (
                <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-primary/20 text-primary">
                  <X className="w-3 h-3" />
                  <span className="text-xs font-medium">불일치</span>
                </div>
              )}
            </div>
          </div>

          {/* Expression Details */}
          <div className="pt-3 border-t border-border">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>얼굴: <span className="text-foreground font-medium">{faceExpression}</span></span>
              <span>탈: <span className="text-foreground font-medium">{maskExpression}</span></span>
            </div>
          </div>
        </div>

        {/* Expandable Details */}
        {maskInfo && (
          <Collapsible open={isOpen} onOpenChange={setIsOpen}>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-between hover:bg-secondary/50 -mx-2 px-2"
              >
                <span className="text-sm font-medium">탈 상세 정보</span>
                <ChevronDown 
                  className={cn(
                    "w-4 h-4 transition-transform duration-200",
                    isOpen && "rotate-180"
                  )} 
                />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-3">
              {/* Description */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-primary" />
                  탈 소개
                </h4>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {maskInfo.description}
                </p>
              </div>

              {/* Dance Role - One line */}
              {maskInfo.danceRole && (
                <p className="text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">역할:</span> {maskInfo.danceRole}
                </p>
              )}

              {/* Character Quote */}
              {maskInfo.quote && (
                <div className="bg-secondary/50 rounded-lg p-3 border-l-2 border-primary">
                  <p className="text-sm text-foreground italic">
                    "{maskInfo.quote}"
                  </p>
                </div>
              )}

              {/* Wiki Link */}
              {maskInfo.wikiUrl && (
                <div className="pt-2">
                  <a
                    href={maskInfo.wikiUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-secondary text-foreground text-xs font-medium hover:bg-secondary/80 transition-colors"
                  >
                    <BookOpen className="w-3 h-3" />
                    더 알아보기
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Fallback if no mask info */}
        {!maskInfo && (
          <p className="text-xs text-muted-foreground text-center py-2">
            이 탈에 대한 상세 정보가 준비 중입니다.
          </p>
        )}
      </div>
    </div>
  );
};
