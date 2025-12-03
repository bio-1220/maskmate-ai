import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { VoteResponse } from '@/types/recommendation';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, BarChart3, RefreshCw } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const API_BASE_URL = 'https://sogangparrot-api.ngrok.app';

const Results: React.FC = () => {
  const [votes, setVotes] = useState<VoteResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchVotes = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/votes`);
      if (response.ok) {
        const data: VoteResponse = await response.json();
        setVotes(data);
      }
    } catch (error) {
      console.error('Failed to fetch votes:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchVotes();
  }, []);

  const chartData = votes ? [
    { name: 'Fine-tuned', value: votes.finetuned, color: 'hsl(var(--primary))' },
    { name: 'Baseline', value: votes.baseline, color: 'hsl(var(--gold))' },
  ] : [];

  const finetunedPercent = votes && votes.total > 0 
    ? ((votes.finetuned / votes.total) * 100).toFixed(1) 
    : '0';
  const baselinePercent = votes && votes.total > 0 
    ? ((votes.baseline / votes.total) * 100).toFixed(1) 
    : '0';

  return (
    <div className="min-h-screen bg-background bg-pattern-traditional">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-gold flex items-center justify-center">
              <span className="text-primary-foreground font-serif font-bold">탈</span>
            </div>
            <h1 className="font-serif text-xl font-semibold text-foreground">투표 결과</h1>
          </div>
          <Link to="/">
            <Button variant="outline" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              추천 받기
            </Button>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 mb-2">
              <BarChart3 className="w-6 h-6 text-primary" />
              <h2 className="font-serif text-3xl font-bold text-foreground">
                모델 비교 투표 결과
              </h2>
            </div>
            <p className="text-muted-foreground">
              Fine-tuned 모델 vs Baseline 모델
            </p>
          </div>

          {isLoading ? (
            <div className="flex justify-center py-12">
              <RefreshCw className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : votes && votes.total > 0 ? (
            <>
              {/* Pie Chart */}
              <Card className="p-6 mb-8">
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={chartData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <Card className="p-6 text-center bg-primary/5 border-primary/20">
                  <p className="text-sm text-muted-foreground mb-1">Fine-tuned Model</p>
                  <p className="text-4xl font-bold text-primary">{votes.finetuned}</p>
                  <p className="text-sm text-muted-foreground mt-1">{finetunedPercent}%</p>
                </Card>
                <Card className="p-6 text-center bg-gold/5 border-gold/20">
                  <p className="text-sm text-muted-foreground mb-1">Baseline Model</p>
                  <p className="text-4xl font-bold text-gold">{votes.baseline}</p>
                  <p className="text-sm text-muted-foreground mt-1">{baselinePercent}%</p>
                </Card>
              </div>

              {/* Total */}
              <Card className="p-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">총 투표 수</p>
                <p className="text-5xl font-bold text-foreground">{votes.total}</p>
              </Card>
            </>
          ) : (
            <Card className="p-12 text-center">
              <p className="text-muted-foreground">아직 투표 데이터가 없습니다.</p>
              <Link to="/" className="mt-4 inline-block">
                <Button variant="traditional">
                  추천 받고 투표하기
                </Button>
              </Link>
            </Card>
          )}

          {/* Refresh Button */}
          <div className="text-center mt-8">
            <Button variant="outline" onClick={fetchVotes} disabled={isLoading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              새로고침
            </Button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Fine-tuned vs Baseline 모델 성능 비교 실험</p>
        </div>
      </footer>
    </div>
  );
};

export default Results;
