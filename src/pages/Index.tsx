import React, { useState } from 'react';
import { ImageUploader } from '@/components/ImageUploader';
import { MaskCard } from '@/components/MaskCard';
import { LoadingState } from '@/components/LoadingState';
import { Button } from '@/components/ui/button';
import { Sparkles, Info, Github } from 'lucide-react';
import { RecommendationResponse, EXPRESSION_LABELS } from '@/types/recommendation';
import { toast } from '@/hooks/use-toast';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Index: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<RecommendationResponse | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
  };

  const handleRecommend = async () => {
    if (!selectedFile) {
      toast({
        title: "이미지를 선택해주세요",
        description: "얼굴 사진을 먼저 업로드해야 합니다.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('추천 요청에 실패했습니다.');
      }

      const data: RecommendationResponse = await response.json();
      setResult(data);
      
      toast({
        title: "추천 완료!",
        description: `당신의 표정: ${EXPRESSION_LABELS[data.face_expression] || data.face_expression}`,
      });
    } catch (error) {
      console.error('Recommendation error:', error);
      toast({
        title: "오류가 발생했습니다",
        description: "백엔드 서버가 실행 중인지 확인해주세요.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background bg-pattern-traditional">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-gold flex items-center justify-center">
              <span className="text-primary-foreground font-serif font-bold">탈</span>
            </div>
            <h1 className="font-serif text-xl font-semibold text-foreground">탈 추천</h1>
          </div>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className="p-2 rounded-lg hover:bg-secondary transition-colors"
          >
            <Github className="w-5 h-5 text-muted-foreground" />
          </a>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-16 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
        <div className="container mx-auto px-4 relative">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="font-serif text-4xl md:text-5xl font-bold text-foreground mb-4">
              당신에게 어울리는
              <br />
              <span className="text-gradient-traditional">한국 전통 탈</span>을 찾아드립니다
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              AI가 당신의 얼굴과 표정을 분석하여
              <br />
              가장 잘 어울리는 전통 탈을 추천해드립니다
            </p>
            
            {/* Info Box */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 text-sm text-muted-foreground">
              <Info className="w-4 h-4" />
              <span>ResNet18 기반 표정 분석 + 임베딩 유사도 매칭</span>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section className="py-8">
        <div className="container mx-auto px-4">
          <div className="max-w-md mx-auto">
            <ImageUploader onImageSelect={handleImageSelect} isLoading={isLoading} />
            
            {selectedFile && !isLoading && (
              <div className="mt-6 animate-fade-up">
                <Button 
                  variant="traditional" 
                  size="xl" 
                  className="w-full"
                  onClick={handleRecommend}
                >
                  <Sparkles className="w-5 h-5" />
                  어울리는 탈 추천받기
                </Button>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Loading State */}
      {isLoading && (
        <section className="py-8">
          <div className="container mx-auto px-4">
            <LoadingState />
          </div>
        </section>
      )}

      {/* Results Section */}
      {result && !isLoading && (
        <section className="py-12">
          <div className="container mx-auto px-4">
            <div className="text-center mb-8">
              <h3 className="font-serif text-2xl font-semibold text-foreground mb-2">
                추천 결과
              </h3>
              <p className="text-muted-foreground">
                감지된 표정: <span className="text-primary font-medium">{EXPRESSION_LABELS[result.face_expression] || result.face_expression}</span>
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              {result.recommendations.map((rec, index) => (
                <MaskCard
                  key={rec.mask_path}
                  rank={index + 1}
                  maskName={rec.mask_name}
                  maskImage={`${API_BASE_URL}/masks/${encodeURIComponent(rec.mask_path)}`}
                  cosineSimilarity={rec.cosine_similarity}
                  expressionMatch={rec.expression_match}
                  combinedScore={rec.combined_score}
                  faceExpression={EXPRESSION_LABELS[result.face_expression] || result.face_expression}
                  maskExpression={EXPRESSION_LABELS[rec.mask_expression] || rec.mask_expression}
                />
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="py-8 border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>PyTorch ResNet18 기반 감정 분석 모델을 활용한 한국 전통 탈 추천 시스템</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
