import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Copy, Check, Undo2, Redo2 } from 'lucide-react';
import { MermaidCodeTab } from '../types';

interface MermaidCodeEditorProps {
  tabs: MermaidCodeTab[];
  activeTabId: number | null;
  onTabChange: (tabId: number) => void;
  onCodeChange: (tabId: number, code: string) => void;
  onRender: (code: string) => void;
}

const MermaidCodeEditor: React.FC<MermaidCodeEditorProps> = ({
  tabs,
  activeTabId,
  onTabChange,
  onCodeChange,
  onRender,
}) => {
  const [editedCode, setEditedCode] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef<number>(-1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 当切换标签时，更新编辑的代码和撤销历史
  useEffect(() => {
    const activeTab = tabs.find((tab) => tab.id === activeTabId);
    if (activeTab) {
      setEditedCode(activeTab.code);
      // 重置撤销历史
      historyRef.current = [activeTab.code];
      historyIndexRef.current = 0;
    }
  }, [activeTabId, tabs]);

  // 保存到历史记录
  const saveToHistory = (code: string) => {
    // 移除当前位置之后的历史（如果有重做操作）
    historyRef.current = historyRef.current.slice(0, historyIndexRef.current + 1);
    // 添加新状态
    historyRef.current.push(code);
    historyIndexRef.current = historyRef.current.length - 1;
    // 限制历史记录长度（最多50条）
    if (historyRef.current.length > 50) {
      historyRef.current.shift();
      historyIndexRef.current--;
    }
  };

  // 撤销
  const handleUndo = () => {
    if (historyIndexRef.current > 0) {
      historyIndexRef.current--;
      const previousCode = historyRef.current[historyIndexRef.current];
      setEditedCode(previousCode);
      if (activeTabId !== null) {
        onCodeChange(activeTabId, previousCode);
      }
    }
  };

  // 重做
  const handleRedo = () => {
    if (historyIndexRef.current < historyRef.current.length - 1) {
      historyIndexRef.current++;
      const nextCode = historyRef.current[historyIndexRef.current];
      setEditedCode(nextCode);
      if (activeTabId !== null) {
        onCodeChange(activeTabId, nextCode);
      }
    }
  };

  const handleCodeChange = (value: string) => {
    setEditedCode(value);
    if (activeTabId !== null) {
      onCodeChange(activeTabId, value);
      saveToHistory(value);
    }
  };

  const handleRender = () => {
    if (editedCode.trim()) {
      onRender(editedCode);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(editedCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const activeTab = tabs.find((tab) => tab.id === activeTabId);

  if (tabs.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400">
        <div className="text-center">
          <p className="text-sm">No diagram code available</p>
          <p className="text-xs mt-1 text-slate-500">Start chatting to generate diagrams</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-900">
      {/* Tabs */}
      <div className="flex gap-2 px-4 pt-4 pb-2 border-b border-slate-800 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-3 py-2 rounded-t-lg text-sm font-medium transition-colors whitespace-nowrap ${
              activeTabId === tab.id
                ? 'bg-slate-800 text-slate-100 border-t border-l border-r border-slate-700'
                : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800/70 hover:text-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Editor Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-slate-800/30">
          <div className="text-xs text-slate-400">
            {activeTab && `Editing: ${activeTab.label}`}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleUndo}
              disabled={historyIndexRef.current <= 0}
              className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-300 rounded border border-slate-700 flex items-center gap-1.5 transition-colors"
              title="Undo (Ctrl+Z)"
            >
              <Undo2 className="w-3 h-3" />
            </button>
            <button
              onClick={handleRedo}
              disabled={historyIndexRef.current >= historyRef.current.length - 1}
              className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-300 rounded border border-slate-700 flex items-center gap-1.5 transition-colors"
              title="Redo (Ctrl+Y)"
            >
              <Redo2 className="w-3 h-3" />
            </button>
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 flex items-center gap-1.5 transition-colors"
              title="Copy code"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  Copy
                </>
              )}
            </button>
            <button
              onClick={handleRender}
              disabled={!editedCode.trim()}
              className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white rounded flex items-center gap-1.5 transition-colors"
              title="Render diagram"
            >
              <RefreshCw className="w-3 h-3" />
              Render
            </button>
          </div>
        </div>

        {/* Code Editor with Line Numbers */}
        <div className="flex-1 overflow-auto bg-slate-950">
          <div className="flex h-full">
            {/* Line Numbers */}
            <div className="flex-shrink-0 px-3 py-4 text-right text-slate-500 font-mono text-sm select-none border-r border-slate-700">
              {editedCode.split('\n').map((_, index) => (
                <div key={index} className="leading-6">
                  {index + 1}
                </div>
              ))}
            </div>
            {/* Code Textarea */}
            <div className="flex-1 p-4">
              <textarea
                ref={textareaRef}
                value={editedCode}
                onChange={(e) => handleCodeChange(e.target.value)}
                onKeyDown={(e) => {
                  // 支持 Ctrl+Z 撤销和 Ctrl+Y 重做
                  if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                    e.preventDefault();
                    handleUndo();
                  } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                    e.preventDefault();
                    handleRedo();
                  }
                }}
                className="w-full h-full bg-transparent text-slate-100 font-mono text-sm focus:outline-none resize-none"
                placeholder="Mermaid diagram code will appear here..."
                spellCheck={false}
                style={{ lineHeight: '1.5rem' }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MermaidCodeEditor;

