import React from 'react';
import { Check, X, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

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
        "relative bg-card rounded-2xl overflow-hidden shadow-card transition-all duration-500 hover:shadow-glow hover:-translate-y-1",
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
        <h3 className="font-serif text-xl font-semibold text-foreground">
          {maskName}
        </h3>

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
      </div>
    </div>
  );
};
