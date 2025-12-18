import { useEffect, useRef, useState, useLayoutEffect } from 'react';
import mermaid from 'mermaid';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface MermaidRendererProps {
  code: string;
}

/**
 * Simple cleanup: only fix the most common issues that LLM might generate
 * Keep it minimal - let Mermaid handle validation, or show error if code is invalid
 */
const cleanMermaidCode = (code: string): string => {
  let cleaned = code.trim();
  
  // 1. Remove HTML tags (LLM sometimes includes these)
  cleaned = cleaned.replace(/<[^>]+>/g, ' ');
  
  // 2. Fix newlines in node labels (Mermaid doesn't support \n in labels)
  cleaned = cleaned.replace(/\[([^\]]*)\]/g, (match, content) => {
    return '[' + content.replace(/\n/g, ' ').replace(/\s+/g, ' ') + ']';
  });
  
  // 3. Fix newlines in edge labels
  cleaned = cleaned.replace(/\|([^|]*)\|/g, (match, content) => {
    return '|' + content.replace(/\n/g, ' ').replace(/\s+/g, ' ') + '|';
  });
  
  // 4. Normalize whitespace (preserve newlines)
  cleaned = cleaned.replace(/[ \t]+/g, ' ');
  
  return cleaned.trim();
};

const MermaidRenderer: React.FC<MermaidRendererProps> = ({ code }) => {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scaledWrapperRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [scale, setScale] = useState(1);
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 8;
  const SCALE_STEP = 0.25;

  // Initialize mermaid once on mount
  useEffect(() => {
    if (!isInitialized) {
      mermaid.initialize({
        startOnLoad: false, // Important: disable auto-rendering
        theme: 'dark',
        securityLevel: 'loose',
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true,
          curve: 'basis',
        },
      });
      setIsInitialized(true);
    }
  }, [isInitialized]);

  // Render diagram when code changes
  useEffect(() => {
    // Wait for initialization
    if (!isInitialized || !mermaidRef.current) {
      return;
    }

    // Handle empty code
    if (!code || code.trim() === '') {
      mermaidRef.current.innerHTML = '';
      setError(null);
      return;
    }

    // Clear previous content and error
    mermaidRef.current.innerHTML = '';
    setError(null);

    // Clean the Mermaid code to fix common issues
    const cleanedCode = cleanMermaidCode(code);

    // Generate unique ID for this diagram
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;

    // Render the diagram asynchronously
    // First try with cleaned code, if it fails, try with original code
    mermaid
      .render(id, cleanedCode)
      .then((result) => {
        // Check if ref is still valid (component hasn't unmounted)
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = result.svg;
          setError(null);
          // Reset scale when new diagram is rendered
          setScale(1);
        }
      })
      .catch((err) => {
        console.error('Mermaid rendering error with cleaned code:', err);
        console.log('Cleaned code:', cleanedCode.substring(0, 200));
        console.log('Original code:', code.substring(0, 200));
        
        // If cleaned code fails, try original code as fallback
        // But first, do minimal cleaning: just remove HTML tags and fix node labels
        const minimalCleaned = code
          .trim()
          .replace(/<[^>]+>/g, ' ') // Remove HTML tags
          .replace(/\[([^\]]*)\]/g, (match, content) => {
            // Only replace newlines in node labels, preserve everything else
            const cleanedContent = content.replace(/\n/g, ' ').replace(/\s+/g, ' ');
            return '[' + cleanedContent + ']';
          })
          .replace(/\|([^|]*)\|/g, (match, content) => {
            const cleanedContent = content.replace(/\n/g, ' ').replace(/\s+/g, ' ');
            return '|' + cleanedContent + '|';
          });
        
        const fallbackId = `mermaid-fallback-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
        mermaid
          .render(fallbackId, minimalCleaned)
          .then((result) => {
            if (mermaidRef.current) {
              mermaidRef.current.innerHTML = result.svg;
              setError(null);
            }
          })
          .catch((fallbackErr) => {
            console.error('Mermaid rendering error with minimal cleaned code:', fallbackErr);
            // Last resort: try completely original code
            const lastResortId = `mermaid-last-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
            mermaid
              .render(lastResortId, code.trim())
              .then((result) => {
                if (mermaidRef.current) {
                  mermaidRef.current.innerHTML = result.svg;
                  setError(null);
                  setScale(1);
                }
              })
              .catch((lastErr) => {
                console.error('Mermaid rendering error with original code:', lastErr);
                const errorMessage = lastErr instanceof Error ? lastErr.message : 'Failed to render diagram';
                setError(errorMessage);
                if (mermaidRef.current) {
                  mermaidRef.current.innerHTML = '';
                }
              });
          });
      });
  }, [code, isInitialized]);

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + SCALE_STEP, MAX_SCALE));
  };

  const handleZoomOut = () => {
    setScale((prev) => Math.max(prev - SCALE_STEP, MIN_SCALE));
  };

  const handleResetZoom = () => {
    setScale(1);
  };

  // 更新缩放后包装元素的尺寸，确保滚动行为正确
  useLayoutEffect(() => {
    if (mermaidRef.current && scaledWrapperRef.current) {
      const svgElement = mermaidRef.current.querySelector('svg');
      if (svgElement) {
        // 临时移除 scale 来获取 SVG 的真实尺寸
        const originalTransform = scaledWrapperRef.current.style.transform;
        scaledWrapperRef.current.style.transform = 'scale(1)';
        
        // 强制重排以获取真实尺寸
        void scaledWrapperRef.current.offsetHeight;
        
        // 获取 SVG 的真实尺寸
        const rect = svgElement.getBoundingClientRect();
        const svgWidth = rect.width;
        const svgHeight = rect.height;
        
        // 恢复 transform
        scaledWrapperRef.current.style.transform = originalTransform;
        
        // 设置包装元素的尺寸为缩放后的尺寸
        // 这样包装元素的实际占用空间就是缩放后的尺寸，滚动条会正确显示
        scaledWrapperRef.current.style.width = `${svgWidth * scale}px`;
        scaledWrapperRef.current.style.height = `${svgHeight * scale}px`;
      }
    }
  }, [scale, code, isInitialized]);

  return (
    <div className="w-full h-full bg-zinc-900 flex flex-col relative">
      {/* Zoom Controls */}
      {!error && code.trim() && (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-zinc-800/90 backdrop-blur-sm border border-zinc-700 rounded-lg p-1.5">
          <button
            onClick={handleZoomOut}
            disabled={scale <= MIN_SCALE}
            className="p-1.5 text-zinc-300 hover:text-zinc-100 hover:bg-zinc-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="px-2 text-xs text-zinc-400 min-w-[3rem] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            disabled={scale >= MAX_SCALE}
            className="p-1.5 text-zinc-300 hover:text-zinc-100 hover:bg-zinc-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <div className="w-px h-4 bg-zinc-700 mx-1" />
          <button
            onClick={handleResetZoom}
            disabled={scale === 1}
            className="p-1.5 text-zinc-300 hover:text-zinc-100 hover:bg-zinc-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Reset zoom"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Diagram Container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto bg-zinc-900"
        style={{
          position: 'relative',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100%',
            padding: '2rem',
          }}
        >
          <div
            ref={scaledWrapperRef}
            style={{
              transform: `scale(${scale})`,
              transformOrigin: 'center center',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {error ? (
              <div className="text-red-400 p-4 text-sm bg-zinc-800/50 border border-red-900/50 rounded-lg max-w-2xl">
                <div className="font-semibold mb-2">Error rendering diagram:</div>
                <div className="text-red-300">{error}</div>
                <div className="mt-3 text-xs text-zinc-400">
                  The generated Mermaid code may be invalid. Please try rephrasing your request.
                </div>
              </div>
            ) : (
              <div
                ref={mermaidRef}
                className="mermaid-container"
                style={{
                  display: 'inline-block',
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MermaidRenderer;

