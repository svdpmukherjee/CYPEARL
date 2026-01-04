/**
 * EmailList Component - CYPEARL Experiment (UPDATED v2)
 * 
 * CHANGES IN THIS VERSION:
 * - Removed "Evaluation Done" badge (now shown in ReadingPane)
 * - Removed "Continue to Next" button (now shown in ReadingPane)
 * - Clean email list focused on email display only
 */

import React from 'react';
import { Search, Filter, Flag, Trash2, Mail } from 'lucide-react';

const EmailList = ({ emails, selectedId, onSelect, onDone, actionsTaken }) => {
    const [activeTab, setActiveTab] = React.useState('focused');

    return (
        <div className="w-[280px] bg-white border-r border-gray-200 flex flex-col h-full">
            {/* Header */}
            <div className="px-4 py-3 flex items-center justify-between bg-white sticky top-0 z-10">
                <div className="flex items-center space-x-1">
                    <div className="font-bold text-base text-[#252423]">Inbox</div>
                </div>
                <div className="flex items-center space-x-2 text-[#0078d4]">
                    <Filter size={16} className="cursor-pointer hover:text-[#106ebe]" />
                    <span className="text-xs font-semibold cursor-pointer hover:underline">Filter</span>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex text-sm font-medium border-b border-gray-200 mx-4">
                <div
                    className={`mr-4 pb-2 border-b-[3px] cursor-pointer ${activeTab === 'focused' ? 'border-[#0078d4] text-[#252423] font-semibold' : 'border-transparent text-[#605e5c] hover:text-[#252423]'}`}
                    onClick={() => setActiveTab('focused')}
                >
                    Focused
                </div>
                <div
                    className={`pb-2 border-b-[3px] cursor-pointer ${activeTab === 'other' ? 'border-[#0078d4] text-[#252423] font-semibold' : 'border-transparent text-[#605e5c] hover:text-[#252423]'}`}
                    onClick={() => setActiveTab('other')}
                >
                    Other
                </div>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto">
                {activeTab === 'focused' && emails && emails.length > 0 ? (
                    emails.map((email, index) => (
                        <EmailListItem
                            key={email.id}
                            sender={email.sender_name}
                            subject={email.subject}
                            preview={email.body
                                .replace(/<[^>]*>?/gm, '')
                                .replace(/\s+/g, ' ')
                                .trim()
                                .substring(0, 30)
                            }
                            time={new Date(email.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            initials={email.sender_name.charAt(0)}
                            selected={email.id === selectedId}
                            read={email.is_read}
                            onClick={() => onSelect(email.id)}
                        />
                    ))
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-[#605e5c] p-8 text-center">
                        <div className="mb-2 font-semibold">All caught up</div>
                        <div className="text-xs">Nothing in {activeTab === 'focused' ? 'Focused' : 'Other'}</div>
                    </div>
                )}
            </div>
        </div>
    );
};

const EmailListItem = ({
    sender,
    subject,
    preview,
    time,
    initials,
    selected,
    read,
    onClick
}) => {
    return (
        <div
            className={`relative group px-4 py-3 cursor-pointer border-b border-gray-100 ${selected
                    ? 'bg-[#c7e0f4] border-l-[4px] border-l-[#0078d4] pl-[12px]'
                    : read
                        ? 'bg-white hover:bg-[#f3f2f1] border-l-[4px] border-l-transparent pl-[12px]'
                        : 'bg-white hover:bg-[#f3f2f1] border-l-[4px] border-l-[#0078d4] pl-[12px]'
                }`}
            onClick={onClick}
        >
            <div className="flex items-start space-x-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${selected ? 'bg-[#0078d4] text-white' : 'bg-[#f3f2f1] text-[#605e5c]'
                    }`}>
                    {initials}
                </div>
                <div className="flex-1 min-w-0">
                    {/* Top row: Sender name and time */}
                    <div className="flex justify-between items-center mb-0.5">
                        <span className={`truncate text-sm ${read ? 'font-normal text-[#252423]' : 'font-bold text-[#252423]'
                            }`}>
                            {sender}
                        </span>
                        <span className={`text-xs whitespace-nowrap ml-2 shrink-0 ${read ? 'text-[#605e5c]' : 'text-[#0078d4] font-semibold'
                            }`}>
                            {time}
                        </span>
                    </div>

                    {/* Subject line */}
                    <div className={`text-sm truncate mb-0.5 ${read ? 'font-normal text-[#252423]' : 'font-bold text-[#0078d4]'
                        }`}>
                        {subject}
                    </div>

                    {/* Preview text */}
                    <div className="text-xs text-[#605e5c] line-clamp-2 leading-relaxed break-words">
                        {preview}
                    </div>
                </div>
            </div>

            {/* Hover Actions (non-functional, for realism) */}
            <div className="absolute top-3 right-2 hidden group-hover:flex items-center space-x-1 bg-white/80 backdrop-blur-sm pl-2 rounded-l">
                <div className="p-1.5 hover:bg-[#edebe9] rounded text-[#605e5c] hover:text-[#252423]" title="Delete">
                    <Trash2 size={16} strokeWidth={1.5} />
                </div>
                <div className="p-1.5 hover:bg-[#edebe9] rounded text-[#605e5c] hover:text-[#252423]" title="Mark as read">
                    <Mail size={16} strokeWidth={1.5} />
                </div>
                <div className="p-1.5 hover:bg-[#edebe9] rounded text-[#605e5c] hover:text-[#252423]" title="Flag">
                    <Flag size={16} strokeWidth={1.5} />
                </div>
            </div>
        </div>
    );
};

export default EmailList;