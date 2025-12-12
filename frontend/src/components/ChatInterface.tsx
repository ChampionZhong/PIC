import { useState, useRef, useEffect } from 'react';
import { Send, User, Bot } from 'lucide-react';
import { Message } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void;
  onMermaidCodeChange: (code: string) => void;
  currentMermaidCode: string;
  inputDraft: string;
  onInputDraftChange: (draft: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  messages,
  setMessages,
  onMermaidCodeChange,
  currentMermaidCode,
  inputDraft,
  onInputDraftChange,
}) => {
  const [input, setInput] = useState(inputDraft);
  const [isLoading, setIsLoading] = useState(false);
  
  // Sync input with draft prop when it changes (e.g., when switching conversations)
  useEffect(() => {
    setInput(inputDraft);
  }, [inputDraft]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const parseResponse = (response: string): { analysis: string; mermaidCode: string } => {
    // Extract analysis block
    const analysisMatch = response.match(/\[\[ANALYSIS_START\]\](.*?)\[\[ANALYSIS_END\]\]/s);
    const analysis = analysisMatch ? analysisMatch[1].trim() : response;

    // Extract mermaid block
    const mermaidMatch = response.match(/\[\[MERMAID_START\]\](.*?)\[\[MERMAID_END\]\]/s);
    const mermaidCode = mermaidMatch ? mermaidMatch[1].trim() : '';

    return { analysis, mermaidCode };
  };

  // Extract analysis text from message content (for display purposes)
  // Removes mermaid code blocks to show only the analysis text
  const extractAnalysisText = (content: string): string => {
    // If content contains mermaid code block, extract only the text before it
    const mermaidBlockMatch = content.match(/```mermaid[\s\S]*?```/);
    if (mermaidBlockMatch && mermaidBlockMatch.index !== undefined) {
      return content.substring(0, mermaidBlockMatch.index).trim();
    }
    return content;
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    // 方案2：构建用户消息，如果有当前 mermaid 代码，附加到消息中
    let userMessageContent = input.trim();
    if (currentMermaidCode && currentMermaidCode.trim()) {
      userMessageContent += `\n\n当前图表:\n\`\`\`mermaid\n${currentMermaidCode}\n\`\`\``;
    }

    // 保存到历史记录时，只保存用户原始输入（不包含图表代码）
    const userMessageForHistory: Message = {
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessageForHistory]);
    setInput('');
    onInputDraftChange(''); // 清空草稿
    setIsLoading(true);

    try {
      // Build history for API（历史记录中包含完整的助手消息，包括 mermaid 代码）
      const history = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessageContent, // 发送包含当前图表的完整消息
          history: history,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const { analysis, mermaidCode } = parseResponse(data.response);

      // 方案1：将 mermaid 代码包含在助手消息中（保存到历史记录）
      let assistantMessageContent = analysis;
      if (mermaidCode && mermaidCode.trim()) {
        assistantMessageContent += `\n\n\`\`\`mermaid\n${mermaidCode}\n\`\`\``;
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: assistantMessageContent, // 包含 mermaid 代码的完整内容
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Update mermaid code if found
      if (mermaidCode) {
        onMermaidCodeChange(mermaidCode);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-100">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
              <div className="text-center">
                <Bot className="w-12 h-12 mx-auto mb-4 text-slate-500" />
                <p className="text-slate-400">Start a conversation to generate research pipeline diagrams</p>
              </div>
            </div>
          )}
          {messages.map((message, index) => {
            // 显示时只显示分析文本（不显示 mermaid 代码块）
            const displayContent = message.role === 'assistant' 
              ? extractAnalysisText(message.content) 
              : message.content;

            return (
              <div
                key={index}
                className={`flex gap-4 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-slate-400" />
                  </div>
                )}
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-slate-800 text-slate-100 border border-slate-700'
                      : 'bg-slate-800/50 text-slate-200 border border-slate-800'
                  }`}
                >
                  <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                    {displayContent}
                  </div>
                </div>
                {message.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                    <User className="w-4 h-4 text-slate-400" />
                  </div>
                )}
              </div>
            );
          })}
          {isLoading && (
            <div className="flex gap-4 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                <Bot className="w-4 h-4 text-slate-400" />
              </div>
              <div className="bg-slate-800/50 border border-slate-800 rounded-lg px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto p-4">
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                const newValue = e.target.value;
                setInput(newValue);
                onInputDraftChange(newValue); // 实时保存草稿
              }}
              onKeyPress={handleKeyPress}
              placeholder="Describe your research pipeline idea..."
              className="flex-1 bg-slate-800/50 border border-slate-700 text-slate-100 placeholder:text-slate-500 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-slate-600 focus:border-slate-600 transition-all"
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
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-200 border border-slate-700 rounded-lg px-4 py-3 flex items-center gap-2 transition-all hover:border-slate-600"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;

