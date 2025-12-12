import { useState, useCallback, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import CanvasPanel from './components/CanvasPanel';
import MermaidCodeEditor from './components/MermaidCodeEditor';
import ConversationTabs from './components/ConversationTabs';
import DataImportExport from './components/DataImportExport';
import { Conversation, Message, MermaidCodeTab } from './types';

const STORAGE_KEY_CONVERSATIONS = 'research-copilot-conversations';
const STORAGE_KEY_ACTIVE_CONVERSATION = 'research-copilot-active-conversation';

function App() {
  // Load conversations from localStorage
  const loadConversationsFromStorage = (): Conversation[] => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_CONVERSATIONS);
      if (stored) {
        const conversations = JSON.parse(stored);
        // 确保旧数据兼容性：为没有inputDraft的对话添加默认值
        return conversations.map((conv: Conversation) => ({
          ...conv,
          inputDraft: conv.inputDraft || '',
        }));
      }
    } catch (error) {
      console.error('Failed to load conversations from localStorage:', error);
    }
    return [];
  };

  const loadActiveConversationFromStorage = (): string | null => {
    try {
      return localStorage.getItem(STORAGE_KEY_ACTIVE_CONVERSATION);
    } catch (error) {
      console.error('Failed to load active conversation from localStorage:', error);
    }
    return null;
  };

  // Initialize with default conversation if none exist
  const initializeConversations = (): Conversation[] => {
    const stored = loadConversationsFromStorage();
    if (stored.length === 0) {
      const defaultConv: Conversation = {
        id: `conv-${Date.now()}`,
        title: 'New Conversation',
        messages: [],
        mermaidTabs: [],
        activeTabId: null,
        tabCounter: 1,
        currentCode: '',
        inputDraft: '',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      return [defaultConv];
    }
    return stored;
  };

  const [conversations, setConversations] = useState<Conversation[]>(initializeConversations);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    loadActiveConversationFromStorage() || conversations[0]?.id || null
  );

  // Get current active conversation
  const activeConversation = conversations.find((c) => c.id === activeConversationId) || conversations[0];

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_CONVERSATIONS, JSON.stringify(conversations));
    } catch (error) {
      console.error('Failed to save conversations to localStorage:', error);
    }
  }, [conversations]);

  // Save active conversation ID to localStorage
  useEffect(() => {
    try {
      if (activeConversationId) {
        localStorage.setItem(STORAGE_KEY_ACTIVE_CONVERSATION, activeConversationId);
      }
    } catch (error) {
      console.error('Failed to save active conversation to localStorage:', error);
    }
  }, [activeConversationId]);

  // Update conversation helper
  const updateConversation = (id: string, updates: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id
          ? { ...conv, ...updates, updatedAt: Date.now() }
          : conv
      )
    );
  };

  // Create new conversation
  const handleNewConversation = () => {
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: 'New Conversation',
      messages: [],
      mermaidTabs: [],
      activeTabId: null,
      tabCounter: 1,
      currentCode: '',
      inputDraft: '',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations((prev) => [...prev, newConv]);
    setActiveConversationId(newConv.id);
  };

  // Switch conversation
  const handleConversationChange = (id: string) => {
    setActiveConversationId(id);
  };

  // Delete conversation
  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => {
      const filtered = prev.filter((conv) => conv.id !== id);
      // If deleted conversation was active, switch to first available
      if (activeConversationId === id && filtered.length > 0) {
        setActiveConversationId(filtered[0].id);
      } else if (filtered.length === 0) {
        // If no conversations left, create a new one
        handleNewConversation();
        return prev; // Don't delete, let handleNewConversation handle it
      }
      return filtered;
    });
  };

  // Rename conversation
  const handleRenameConversation = (id: string, newTitle: string) => {
    updateConversation(id, { title: newTitle });
  };

  // Handle import conversations
  const handleImportConversations = (imported: Conversation[]) => {
    setConversations(imported);
    // Set first conversation as active if available
    if (imported.length > 0) {
      setActiveConversationId(imported[0].id);
    }
  };

  // Update messages
  const handleMessagesChange = useCallback(
    (updater: Message[] | ((prev: Message[]) => Message[])) => {
      if (!activeConversation) return;
      const newMessages = typeof updater === 'function' ? updater(activeConversation.messages) : updater;
      updateConversation(activeConversation.id, { messages: newMessages });
      
      // Update title from first user message if title is still default
      if (activeConversation.title === 'New Conversation' && newMessages.length > 0) {
        const firstUserMessage = newMessages.find((m) => m.role === 'user');
        if (firstUserMessage) {
          const title = firstUserMessage.content.slice(0, 30).trim() || 'New Conversation';
          updateConversation(activeConversation.id, { title });
        }
      }
    },
    [activeConversation]
  );

  // Handle mermaid code change
  const handleMermaidCodeChange = useCallback(
    (code: string) => {
      if (!activeConversation || !code.trim()) return;

      const newTab: MermaidCodeTab = {
        id: activeConversation.tabCounter,
        label: `Diagram ${activeConversation.tabCounter}`,
        code: code,
      };

      updateConversation(activeConversation.id, {
        mermaidTabs: [...activeConversation.mermaidTabs, newTab],
        activeTabId: newTab.id,
        tabCounter: activeConversation.tabCounter + 1,
        currentCode: code,
      });
    },
    [activeConversation]
  );

  // Handle mermaid tab change
  const handleTabChange = (tabId: number) => {
    if (!activeConversation) return;
    const tab = activeConversation.mermaidTabs.find((t) => t.id === tabId);
    if (tab) {
      updateConversation(activeConversation.id, {
        activeTabId: tabId,
        currentCode: tab.code,
      });
    }
  };

  // Handle code edit
  const handleCodeChange = (tabId: number, code: string) => {
    if (!activeConversation) return;
    const updatedTabs = activeConversation.mermaidTabs.map((tab) =>
      tab.id === tabId ? { ...tab, code } : tab
    );
    updateConversation(activeConversation.id, { mermaidTabs: updatedTabs });
  };

  // Handle render
  const handleRender = useCallback((code: string) => {
    if (!activeConversation) {
      console.warn('No active conversation for render');
      return;
    }
    
    // Update current code to trigger re-render
    const updatedTabs = activeConversation.activeTabId !== null
      ? activeConversation.mermaidTabs.map((tab) =>
          tab.id === activeConversation.activeTabId ? { ...tab, code } : tab
        )
      : activeConversation.mermaidTabs;
    
    updateConversation(activeConversation.id, { 
      currentCode: code,
      mermaidTabs: updatedTabs,
    });
  }, [activeConversation]);

  // Get current edited code for context
  const getCurrentEditedCode = (): string => {
    if (!activeConversation) return '';
    if (activeConversation.activeTabId !== null) {
      const tab = activeConversation.mermaidTabs.find((t) => t.id === activeConversation.activeTabId);
      return tab ? tab.code : activeConversation.currentCode;
    }
    return activeConversation.currentCode;
  };

  if (!activeConversation) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-slate-900 text-slate-400">
        <p>No conversation available</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-900">
      {/* Left Panel - Chat Interface */}
      <div className="w-1/2 border-r border-slate-800 flex flex-col">
        {/* Import/Export Controls */}
        <DataImportExport
          conversations={conversations}
          onImport={handleImportConversations}
        />
        {/* Conversation Tabs */}
        <ConversationTabs
          conversations={conversations}
          activeConversationId={activeConversationId}
          onConversationChange={handleConversationChange}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onRenameConversation={handleRenameConversation}
        />
        {/* Chat Interface */}
        <div className="flex-1 min-h-0">
          <ChatInterface
            messages={activeConversation.messages}
            setMessages={handleMessagesChange}
            onMermaidCodeChange={handleMermaidCodeChange}
            currentMermaidCode={getCurrentEditedCode()}
            inputDraft={activeConversation.inputDraft || ''}
            onInputDraftChange={(draft) => updateConversation(activeConversation.id, { inputDraft: draft })}
          />
        </div>
      </div>

      {/* Right Panel - Split into Code Editor (top) and Canvas Panel (bottom) */}
      <div className="w-1/2 flex flex-col">
        {/* Top: Code Editor */}
        <div className="h-1/2 border-b border-slate-800">
          <MermaidCodeEditor
            tabs={activeConversation.mermaidTabs}
            activeTabId={activeConversation.activeTabId}
            onTabChange={handleTabChange}
            onCodeChange={handleCodeChange}
            onRender={handleRender}
          />
        </div>

        {/* Bottom: Canvas Panel (Blueprint & Rendering) */}
        <div className="h-1/2">
          <CanvasPanel code={activeConversation.currentCode || ''} />
        </div>
      </div>
    </div>
  );
}

export default App;
