import React, { useState } from 'react';
import { MaskRecommendation, VoteResponse } from '@/types/recommendation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Check, Vote } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

interface VotingSectionProps {
  finetunedTop1: MaskRecommendation;
  baselineTop1: MaskRecommendation;
  apiBaseUrl: string;
}

export const VotingSection: React.FC<VotingSectionProps> = ({
  finetunedTop1,
  baselineTop1,
  apiBaseUrl,
}) => {
  const [hasVoted, setHasVoted] = useState(false);
  const [isVoting, setIsVoting] = useState(false);
  const [votedFor, setVotedFor] = useState<'finetuned' | 'baseline' | null>(null);

  const handleVote = async (choice: 'finetuned' | 'baseline') => {
    if (hasVoted || isVoting) return;
    
    setIsVoting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/vote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vote: choice }),
      });
      
      if (!response.ok) throw new Error('투표 실패');
      
      const result: VoteResponse = await response.json();
      setHasVoted(true);
      setVotedFor(choice);
      
      toast({
        title: "투표 완료!",
        description: `총 ${result.total}명이 투표했습니다.`,
      });
    } catch (error) {
      toast({
        title: "투표 오류",
        description: "다시 시도해주세요.",
        variant: "destructive",
      });
    } finally {
      setIsVoting(false);
    }
  };

  const cleanMaskName = (name: string) => name.replace(/_/g, ' ').replace(/\.(jpg|png|jpeg)$/i, '');

  return (
    <section className="py-12 border-t border-border/50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-2">
            <Vote className="w-5 h-5 text-primary" />
            <h3 className="font-serif text-xl font-semibold text-foreground">
              모델 성능 비교 투표
            </h3>
          </div>
          <p className="text-sm text-muted-foreground">
            어떤 추천이 더 잘 어울린다고 생각하시나요?
          </p>
        </div>

        <div className="max-w-3xl mx-auto grid grid-cols-2 gap-6">
          {/* Fine-tuned Model */}
          <Card 
            className={`p-4 cursor-pointer transition-all hover:shadow-lg ${
              votedFor === 'finetuned' ? 'ring-2 ring-primary bg-primary/5' : ''
            } ${hasVoted && votedFor !== 'finetuned' ? 'opacity-50' : ''}`}
            onClick={() => handleVote('finetuned')}
          >
            <div className="text-center mb-3">
              <span className="text-xs font-medium px-2 py-1 rounded-full bg-primary/10 text-primary">
                Fine-tuned Model
              </span>
            </div>
            <div className="aspect-square rounded-lg overflow-hidden bg-secondary/30 mb-3">
              <img
                src={`${apiBaseUrl}/masks/${encodeURIComponent(finetunedTop1.mask_path)}`}
                alt={finetunedTop1.mask_name}
                className="w-full h-full object-contain"
              />
            </div>
            <p className="text-center font-medium text-foreground">
              {cleanMaskName(finetunedTop1.mask_name)}
            </p>
            {votedFor === 'finetuned' && (
              <div className="flex justify-center mt-2">
                <Check className="w-5 h-5 text-primary" />
              </div>
            )}
          </Card>

          {/* Baseline Model */}
          <Card 
            className={`p-4 cursor-pointer transition-all hover:shadow-lg ${
              votedFor === 'baseline' ? 'ring-2 ring-gold bg-gold/5' : ''
            } ${hasVoted && votedFor !== 'baseline' ? 'opacity-50' : ''}`}
            onClick={() => handleVote('baseline')}
          >
            <div className="text-center mb-3">
              <span className="text-xs font-medium px-2 py-1 rounded-full bg-gold/10 text-gold">
                Baseline (ImageNet)
              </span>
            </div>
            <div className="aspect-square rounded-lg overflow-hidden bg-secondary/30 mb-3">
              <img
                src={`${apiBaseUrl}/masks/${encodeURIComponent(baselineTop1.mask_path)}`}
                alt={baselineTop1.mask_name}
                className="w-full h-full object-contain"
              />
            </div>
            <p className="text-center font-medium text-foreground">
              {cleanMaskName(baselineTop1.mask_name)}
            </p>
            {votedFor === 'baseline' && (
              <div className="flex justify-center mt-2">
                <Check className="w-5 h-5 text-gold" />
              </div>
            )}
          </Card>
        </div>

        {hasVoted && (
          <p className="text-center text-sm text-muted-foreground mt-6 animate-fade-up">
            투표해주셔서 감사합니다! 전체 결과는 상단 메뉴에서 확인하세요.
          </p>
        )}
      </div>
    </section>
  );
};
