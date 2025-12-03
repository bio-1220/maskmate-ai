import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  isLoading?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelect, isLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFile = useCallback((file: File) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      onImageSelect(file);
    }
  }, [onImageSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const clearPreview = useCallback(() => {
    setPreview(null);
  }, []);

  return (
    <div className="w-full max-w-md mx-auto">
      {!preview ? (
        <label
          className={cn(
            "flex flex-col items-center justify-center w-full h-64 rounded-2xl cursor-pointer transition-all duration-300",
            "border-2 border-dashed bg-card/50 backdrop-blur-sm",
            isDragOver 
              ? "border-primary bg-primary/10 scale-[1.02]" 
              : "border-border hover:border-primary/50 hover:bg-card"
          )}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <div className={cn(
              "p-4 rounded-full mb-4 transition-all duration-300",
              isDragOver ? "bg-primary/20" : "bg-secondary"
            )}>
              <Upload className={cn(
                "w-8 h-8 transition-colors",
                isDragOver ? "text-primary" : "text-muted-foreground"
              )} />
            </div>
            <p className="mb-2 text-lg font-medium text-foreground">
              얼굴 사진을 업로드하세요
            </p>
            <p className="text-sm text-muted-foreground">
              드래그 앤 드롭 또는 클릭하여 선택
            </p>
            <p className="mt-2 text-xs text-muted-foreground">
              PNG, JPG, JPEG (최대 10MB)
            </p>
          </div>
          <input 
            type="file" 
            className="hidden" 
            accept="image/*"
            onChange={handleInputChange}
            disabled={isLoading}
          />
        </label>
      ) : (
        <div className="relative animate-scale-in">
          <div className="relative w-full aspect-square rounded-2xl overflow-hidden shadow-card bg-card">
            <img 
              src={preview} 
              alt="업로드된 얼굴" 
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-foreground/20 to-transparent" />
          </div>
          <Button
            variant="destructive"
            size="icon"
            className="absolute -top-2 -right-2 rounded-full shadow-card"
            onClick={clearPreview}
            disabled={isLoading}
          >
            <X className="w-4 h-4" />
          </Button>
          <div className="absolute bottom-4 left-4 right-4">
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-background/80 backdrop-blur-sm">
              <ImageIcon className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium truncate">얼굴 이미지 준비 완료</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
