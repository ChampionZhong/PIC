import { useState } from 'react';
import { Plus, X, Edit2, Check, X as XIcon } from 'lucide-react';
import { Conversation } from '../types';

interface ConversationTabsProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onConversationChange: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
}

const ConversationTabs: React.FC<ConversationTabsProps> = ({
  conversations,
  activeConversationId,
  onConversationChange,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState<string>('');

  const handleStartEdit = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setEditTitle(conversation.title);
  };

  const handleSaveEdit = (id: string) => {
    if (editTitle.trim()) {
      onRenameConversation(id, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  return (
    <div className="flex items-center gap-1 px-2 py-2 border-b border-slate-800 bg-slate-900 overflow-x-auto">
      {conversations.map((conversation) => (
        <div
          key={conversation.id}
          className={`group flex items-center gap-1 px-3 py-1.5 rounded-t-lg text-sm transition-colors ${
            activeConversationId === conversation.id
              ? 'bg-slate-800 text-slate-100'
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800/70 hover:text-slate-300'
          }`}
        >
          {editingId === conversation.id ? (
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSaveEdit(conversation.id);
                  } else if (e.key === 'Escape') {
                    handleCancelEdit();
                  }
                }}
                className="bg-slate-700 text-slate-100 px-2 py-0.5 rounded text-xs min-w-[100px] focus:outline-none focus:ring-1 focus:ring-blue-500"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
              <button
                onClick={() => handleSaveEdit(conversation.id)}
                className="p-0.5 hover:bg-slate-600 rounded"
                title="Save"
              >
                <Check className="w-3 h-3" />
              </button>
              <button
                onClick={handleCancelEdit}
                className="p-0.5 hover:bg-slate-600 rounded"
                title="Cancel"
              >
                <XIcon className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={() => onConversationChange(conversation.id)}
                className="flex-1 text-left truncate max-w-[150px]"
                title={conversation.title}
              >
                {conversation.title}
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleStartEdit(conversation);
                }}
                className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-slate-700 rounded transition-opacity"
                title="Rename"
              >
                <Edit2 className="w-3 h-3" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(conversation.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-red-600 rounded transition-opacity"
                title="Delete"
              >
                <X className="w-3 h-3" />
              </button>
            </>
          )}
        </div>
      ))}
      <button
        onClick={onNewConversation}
        className="px-3 py-1.5 rounded-t-lg text-sm bg-slate-800/50 text-slate-400 hover:bg-slate-800/70 hover:text-slate-300 transition-colors flex items-center gap-1"
        title="New conversation"
      >
        <Plus className="w-4 h-4" />
        <span>New</span>
      </button>
    </div>
  );
};

export default ConversationTabs;

