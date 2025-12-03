export interface MaskRecommendation {
  mask_path: string;
  mask_name: string;
  cosine_similarity: number;
  expression_match: boolean;
  combined_score: number;
  mask_expression: string;
}

export interface RecommendationResponse {
  face_expression: string;
  recommendations: MaskRecommendation[];
}

export const EXPRESSION_LABELS: Record<string, string> = {
  angry: '분노',
  happy: '행복',
  natural: '평온',
  sad: '슬픔',
};
