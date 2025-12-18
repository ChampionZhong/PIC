import { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Sparkles, Palette, Image, CheckCircle2, XCircle, Settings } from 'lucide-react';

export type ChatSender = 'User' | 'Manager' | 'Architect' | 'Director' | 'Painter' | 'Critic';

export interface ChatMessage {
  sender: ChatSender;
  message: string;
  timestamp?: number;
}

interface ChatPanelProps {
  chatHistory: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}

const getSenderIcon = (sender: ChatSender) => {
  switch (sender) {
    case 'Manager':
      return <Settings className="w-4 h-4 text-amber-400" />;
    case 'Architect':
      return <Sparkles className="w-4 h-4 text-indigo-400" />;
    case 'Director':
      return <Palette className="w-4 h-4 text-purple-400" />;
    case 'Painter':
      return <Image className="w-4 h-4 text-cyan-400" />;
    case 'Critic':
      return <CheckCircle2 className="w-4 h-4 text-rose-400" />;
    case 'User':
      return <User className="w-4 h-4 text-slate-400" />;
    default:
      return <Bot className="w-4 h-4 text-slate-400" />;
  }
};

const getSenderColor = (sender: ChatSender) => {
  switch (sender) {
    case 'Manager':
      return 'bg-amber-500/10 border-amber-500/20 text-amber-300';
    case 'Architect':
      return 'bg-indigo-500/10 border-indigo-500/20 text-indigo-300';
    case 'Director':
      return 'bg-purple-500/10 border-purple-500/20 text-purple-300';
    case 'Painter':
      return 'bg-cyan-500/10 border-cyan-500/20 text-cyan-300';
    case 'Critic':
      return 'bg-rose-500/10 border-rose-500/20 text-rose-300';
    case 'User':
      return 'bg-slate-800 text-slate-100 border-slate-700';
    default:
      return 'bg-slate-800/50 text-slate-200 border-slate-800';
  }
};

const ChatPanel: React.FC<ChatPanelProps> = ({ chatHistory, onSendMessage, isLoading = false }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
    // Auto-reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950 text-zinc-100 border-r border-zinc-800">
      {/* Header */}
      <div className="px-6 py-4 border-b border-zinc-800">
        <h2 className="text-lg font-semibold text-zinc-100">Research Studio</h2>
        <p className="text-sm text-zinc-400 mt-1">Multi-Agent Workflow</p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-full space-y-4">
          {chatHistory.length === 0 && (
            <div className="flex items-center justify-center h-full min-h-[400px] text-zinc-400">
              <div className="text-center">
                <Bot className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
                <p className="text-zinc-500">Start by describing your research idea</p>
                <p className="text-sm text-zinc-600 mt-2">The Architect will help you create a diagram</p>
              </div>
            </div>
          )}
          {chatHistory.map((msg, index) => {
            const isUser = msg.sender === 'User';
            const isManager = msg.sender === 'Manager';
            return (
              <div
                key={index}
                className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                    {getSenderIcon(msg.sender)}
                  </div>
                )}
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-3 border ${
                    isUser
                      ? getSenderColor('User')
                      : getSenderColor(msg.sender)
                  } ${isManager ? 'opacity-80 italic' : ''}`}
                >
                  {!isUser && (
                    <div className="text-xs font-medium mb-1.5 opacity-70">
                      {msg.sender}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                    {msg.message}
                  </div>
                </div>
                {isUser && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                    <User className="w-4 h-4 text-zinc-400" />
                  </div>
                )}
              </div>
            );
          })}
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                <Bot className="w-4 h-4 text-zinc-400" />
              </div>
              <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-zinc-800 bg-zinc-950/50 backdrop-blur-sm">
        <div className="p-4">
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                const newValue = e.target.value;
                setInput(newValue);
              }}
              onKeyPress={handleKeyPress}
              placeholder="Describe your research idea..."
              className="flex-1 bg-zinc-900/50 border border-zinc-800 text-zinc-100 placeholder:text-zinc-500 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all text-sm"
              rows={1}
              style={{
                minHeight: '44px',
                maxHeight: '120px',
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white rounded-lg px-4 py-3 flex items-center gap-2 transition-all font-medium"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;

