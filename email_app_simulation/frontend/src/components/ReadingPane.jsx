import React, { useState, useRef, useEffect } from 'react';
import { Reply, ReplyAll, Forward, MoreHorizontal, Trash2, Flag, AlertOctagon, Printer, ExternalLink, Archive, Ban, CheckCircle, Mail } from 'lucide-react';

const ReadingPane = ({ email, onAction, isLatest, actionsTaken, onDone, isFinished }) => {
    const [hoveredLink, setHoveredLink] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
    const hoverStartTime = useRef(null);

    if (isFinished) {
        return (
            <div className="flex-1 bg-white flex items-center justify-center">
                <div className="text-center">
                    <h2 className="text-2xl font-semibold text-[#252423] mb-2">That's all.</h2>
                    <p className="text-[#605e5c] text-lg">Thanks for your time.</p>
                </div>
            </div>
        );
    }

    if (!email) {
        return (
            <div className="flex-1 bg-[#f3f2f1] flex items-center justify-center">
                <div className="text-center">
                    <div className="w-32 h-32 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
                        <Mail size={48} className="text-gray-400" />
                    </div>
                    <p className="text-[#605e5c]">Select an item to read</p>
                </div>
            </div>
        );
    }

    const handleLinkHover = (e, type) => {
        if (e.target.tagName === 'A') {
            const link = e.target.href;

            if (type === 'enter') {
                hoverStartTime.current = Date.now();
                setHoveredLink(link);
                setTooltipPos({ x: e.clientX, y: e.clientY + 20 });
            } else if (type === 'leave') {
                if (hoverStartTime.current) {
                    const duration = Date.now() - hoverStartTime.current;
                    // Only log if duration is significant (> 100ms) to avoid accidental swipes
                    if (duration > 100) {
                        onAction('link_hover', { link, duration });
                    }
                    hoverStartTime.current = null;
                }
                setHoveredLink(null);
            }
        }
    };

    // Update tooltip position on mouse move if hovering
    const handleMouseMove = (e) => {
        if (hoveredLink) {
            setTooltipPos({ x: e.clientX, y: e.clientY + 20 });
        }
    };

    return (
        <div className="flex-1 bg-white h-full flex flex-col overflow-hidden relative" onMouseMove={handleMouseMove}>
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-white sticky top-0 z-10">
                <div className="flex items-center space-x-1">
                    <ActionButton icon={Reply} label="Reply" disabled={!isLatest} />
                    <ActionButton icon={ReplyAll} label="Reply all" disabled={!isLatest} />
                    <ActionButton icon={Forward} label="Forward" disabled={!isLatest} />
                    <div className="h-4 w-px bg-gray-300 mx-2"></div>
                    <ActionButton icon={Trash2} label="Delete" onClick={() => isLatest && onAction('delete')} disabled={!isLatest} />
                    <ActionButton icon={Archive} label="Archive" disabled={!isLatest} />
                    <ActionButton icon={AlertOctagon} label="Report" onClick={() => isLatest && onAction('report')} disabled={!isLatest} />
                </div>
                <div className="flex items-center space-x-1">
                    <ActionButton icon={ExternalLink} disabled={!isLatest} />
                    <ActionButton icon={MoreHorizontal} disabled={!isLatest} />
                </div>
            </div>

            {/* Email Header */}
            <div className="px-8 py-6">
                <div className="flex justify-between items-start mb-6">
                    <h1 className="text-xl font-semibold text-[#252423] leading-tight flex-1 mr-4">{email.subject}</h1>

                    {/* Done Button removed from header */}
                </div>

                <div className="flex items-start justify-between">
                    <div
                        className="flex items-center space-x-3"
                        onMouseEnter={() => {
                            if (isLatest) {
                                hoverStartTime.current = Date.now();
                            }
                        }}
                        onMouseLeave={() => {
                            if (isLatest && hoverStartTime.current) {
                                const duration = Date.now() - hoverStartTime.current;
                                if (duration > 100) {
                                    onAction('sender_hover', { duration });
                                }
                                hoverStartTime.current = null;
                            }
                        }}
                    >
                        <div className="w-12 h-12 rounded-full bg-[#0078d4] flex items-center justify-center text-white font-bold text-lg">
                            {email.sender_name.charAt(0)}
                        </div>
                        <div>
                            <div className="flex items-baseline space-x-2">
                                <span className="font-semibold text-[#252423] text-base">{email.sender_name}</span>
                                <span className="text-xs text-[#605e5c]">&lt;{email.sender_email}&gt;</span>
                            </div>
                            <div className="text-xs text-[#605e5c] mt-0.5">
                                To: You
                            </div>
                        </div>
                    </div>
                    <div className="text-xs text-[#605e5c] flex flex-col items-end">
                        <span>{new Date(email.timestamp).toLocaleString([], { weekday: 'short', year: 'numeric', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                </div>
            </div>

            {/* Email Body */}
            <div className="flex-1 overflow-y-auto px-8 pb-8 flex flex-col">
                <div
                    className="prose max-w-none text-sm text-[#252423] font-sans flex-grow"
                    dangerouslySetInnerHTML={{ __html: email.body }}
                    onClick={(e) => {
                        // Intercept link clicks
                        if (e.target.tagName === 'A') {
                            e.preventDefault();
                            onAction('link_click', { link: e.target.href });
                        }
                    }}
                    onMouseOver={(e) => handleLinkHover(e, 'enter')}
                    onMouseOut={(e) => handleLinkHover(e, 'leave')}
                />

                {/* Decision Buttons - Hide for Email 0 (Welcome) and previous emails */}
                {isLatest && email.order_id !== 0 && (
                    <div className="mt-8 pt-6 border-t border-gray-200">
                        <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
                            <button
                                onClick={() => onAction('safe')}
                                disabled={!isLatest}
                                className={`flex items-center justify-center px-4 py-3 rounded-md transition-colors font-semibold shadow-sm ${isLatest
                                    ? 'bg-green-600 text-white hover:bg-green-700'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                            >
                                <span className="mr-2">✓</span> I believe this email is safe
                            </button>
                            <button
                                onClick={() => onAction('report')}
                                disabled={!isLatest}
                                className={`flex items-center justify-center px-4 py-3 rounded-md transition-colors font-semibold shadow-sm ${isLatest
                                    ? 'bg-red-600 text-white hover:bg-red-700'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                            >
                                <AlertOctagon size={18} className="mr-2" /> I will report this email as suspicious
                            </button>
                            <button
                                onClick={() => onAction('delete')}
                                disabled={!isLatest}
                                className={`flex items-center justify-center px-4 py-3 rounded-md transition-colors font-semibold shadow-sm ${isLatest
                                    ? 'bg-gray-600 text-white hover:bg-gray-700'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                            >
                                <Trash2 size={18} className="mr-2" /> I will delete this email
                            </button>
                            <button
                                onClick={() => onAction('ignore')}
                                disabled={!isLatest}
                                className={`flex items-center justify-center px-4 py-3 rounded-md transition-colors font-semibold shadow-sm ${isLatest
                                    ? 'bg-yellow-500 text-white hover:bg-yellow-600'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                            >
                                <Ban size={18} className="mr-2" /> I will ignore this email
                            </button>
                        </div>
                    </div>
                )}


            </div>

            {/* Custom Link Tooltip */}
            {hoveredLink && (
                <div
                    className="fixed z-50 bg-[#292929] text-white text-xs px-2 py-1 rounded shadow-lg max-w-xs break-all pointer-events-none"
                    style={{
                        left: Math.min(tooltipPos.x, window.innerWidth - 200), // Prevent overflow right
                        top: tooltipPos.y
                    }}
                >
                    {hoveredLink}
                </div>
            )}
        </div>
    );
};

const ActionButton = ({ icon: Icon, label, onClick, disabled }) => {
    return (
        <button
            className={`flex items-center space-x-1.5 px-2 py-1.5 rounded text-sm transition-colors ${disabled
                ? 'text-gray-400 cursor-not-allowed'
                : 'text-[#252423] hover:bg-[#edebe9] cursor-pointer'
                }`}
            onClick={!disabled ? onClick : undefined}
            title={label}
            disabled={disabled}
        >
            <Icon size={16} strokeWidth={1.5} className={disabled ? 'text-gray-400' : 'text-[#0078d4]'} />
            {label && <span className="hidden xl:inline font-medium">{label}</span>}
        </button>
    );
};

export default ReadingPane;
