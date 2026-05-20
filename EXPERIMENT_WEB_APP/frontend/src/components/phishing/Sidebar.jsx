import React from 'react';
import { Inbox, Send, Archive, Trash2, FileText, AlertOctagon, ChevronRight, ChevronDown } from 'lucide-react';

const Sidebar = ({ activeFolder = 'inbox', onFolderSelect, unreadCount, deletedCount }) => {
    return (
        <div className="w-[180px] bg-[#f3f2f1] flex flex-col h-full border-r border-gray-200 text-sm select-none pt-2">
            {/* Favorites Section */}
            <div className="mb-2">
                <div className="flex items-center text-[#605e5c] px-4 py-2 cursor-pointer hover:bg-[#edebe9]">
                    <ChevronDown size={12} className="mr-3" />
                    <span className="font-semibold uppercase text-xs tracking-wider">Favorites</span>
                </div>
                <div className="space-y-[1px]">
                    <SidebarItem icon={Inbox} label="Inbox" count={unreadCount > 0 ? unreadCount : null} active={activeFolder === 'inbox'} onClick={() => onFolderSelect('inbox')} />
                    <SidebarItem icon={Send} label="Sent Items" active={activeFolder === 'sent'} onClick={() => onFolderSelect('sent')} />
                    <SidebarItem icon={FileText} label="Drafts" active={activeFolder === 'drafts'} onClick={() => onFolderSelect('drafts')} />
                </div>
            </div>

            {/* Folders Section */}
            <div className="flex-1 overflow-y-auto">
                <div className="flex items-center text-[#605e5c] px-4 py-2 cursor-pointer hover:bg-[#edebe9]">
                    <ChevronDown size={12} className="mr-3" />
                    <span className="font-semibold uppercase text-xs tracking-wider">Folders</span>
                </div>
                <div className="space-y-[1px]">
                    <SidebarItem icon={Inbox} label="Inbox" count={unreadCount > 0 ? unreadCount : null} active={activeFolder === 'inbox'} onClick={() => onFolderSelect('inbox')} />
                    <SidebarItem icon={FileText} label="Drafts" active={activeFolder === 'drafts'} onClick={() => onFolderSelect('drafts')} />
                    <SidebarItem icon={Archive} label="Archive" active={activeFolder === 'archive'} onClick={() => onFolderSelect('archive')} />
                    <SidebarItem icon={Send} label="Sent Items" active={activeFolder === 'sent'} onClick={() => onFolderSelect('sent')} />
                    <SidebarItem icon={Trash2} label="Deleted Items" count={deletedCount > 0 ? deletedCount : null} active={activeFolder === 'deleted'} onClick={() => onFolderSelect('deleted')} />
                    <SidebarItem icon={AlertOctagon} label="Junk Email" active={activeFolder === 'junk'} onClick={() => onFolderSelect('junk')} />
                </div>
            </div>
        </div>
    );
};

const SidebarItem = ({ icon: Icon, label, count, active, onClick }) => {
    return (
        <div
            className={`flex items-center justify-between px-4 py-[9px] cursor-pointer group ${active ? 'bg-[#c7e0f4] text-black' : 'hover:bg-[#edebe9] text-[#252423]'}`}
            onClick={onClick}
        >
            <div className="flex items-center">
                <Icon size={16} strokeWidth={1.5} className={`mr-3 ${active ? 'text-[#0078d4]' : 'text-[#605e5c] group-hover:text-[#252423]'}`} />
                <span className={`${active ? 'font-semibold' : 'font-normal'}`}>{label}</span>
            </div>
            {count && (
                <span className={`text-xs ${active ? 'text-[#0078d4] font-bold' : 'text-[#0078d4] font-semibold'}`}>
                    {count}
                </span>
            )}
        </div>
    );
};

export default Sidebar;
