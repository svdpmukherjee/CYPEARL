/**
 * ReadingPane Component - CYPEARL Experiment (UPDATED v3 - FIXED)
 * 
 * FIXES IN THIS VERSION:
 * 1. Removed duplicate openEmailSession call (Issue #2) - now handled by EmailApp.jsx
 * 2. Clear hover state before recording link_click (Issue #3) - prevents link_hover on click
 * 
 * FEATURES:
 * - Sender email hidden by default - only name shown
 * - Expandable sender details panel (like Gmail) - tracks sender_click
 * - No fake page overlay - links don't show any feedback
 * - No bonus display - bonus calculated silently in backend
 * - "Evaluation Done" badge in email heading (right side) with pulse
 * - Link clicks tracked silently without visual feedback
 * 
 * Tracks observational data for phishing_study_responses.csv:
 * - dwell_time_ms: Total time email is visible
 * - response_latency_ms: Time from open to action
 * - hovered_link: Whether user hovered over any link
 * - clicked (link): Whether user clicked any link in body
 * - inspected_sender: Whether user clicked to expand sender details (sender_click)
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Reply, ReplyAll, Forward, MoreHorizontal, Trash2, Flag,
    AlertOctagon, Printer, ExternalLink, Archive, Ban, CheckCircle, Mail,
    ChevronDown, ChevronUp, Shield, Lock, X
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// =========================================================================
// SENDER DETAILS PANEL COMPONENT (Gmail-style expandable)
// =========================================================================

const SenderDetailsPanel = ({ email, isExpanded, onToggle, onSenderClick }) => {
    const hasClicked = useRef(false);

    const handleToggle = () => {
        if (!hasClicked.current && !isExpanded) {
            // First time expanding - record the click
            hasClicked.current = true;
            onSenderClick();
        }
        onToggle();
    };

    // Reset when email changes
    useEffect(() => {
        hasClicked.current = false;
    }, [email?.id]);

    // Extract domain from email for mailed-by/signed-by
    const getEmailDomain = (emailAddr) => {
        if (!emailAddr) return 'unknown';
        const match = emailAddr.match(/@([^>]+)/);
        return match ? match[1] : 'unknown';
    };

    const domain = getEmailDomain(email?.sender_email);
    const isInternalDomain = domain.includes('luxconsultancy.com');

    return (
        <div className="relative">
            {/* Clickable "To: me" row with expand arrow */}
            <div
                className="flex items-center text-xs text-[#605e5c] cursor-pointer hover:text-[#252423] group"
                onClick={handleToggle}
            >
                <span>to me</span>
                {isExpanded ? (
                    <ChevronUp size={14} className="ml-1 text-[#605e5c] group-hover:text-[#252423]" />
                ) : (
                    <ChevronDown size={14} className="ml-1 text-[#605e5c] group-hover:text-[#252423]" />
                )}
            </div>

            {/* Expanded Details Panel */}
            {isExpanded && (
                <div className="absolute top-6 left-0 z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-[350px] text-sm">
                    <div className="space-y-2">
                        <div className="flex">
                            <span className="text-[#605e5c] w-20 shrink-0">from:</span>
                            <span className="text-[#252423] font-medium">
                                {email?.sender_name} &lt;{email?.sender_email}&gt;
                            </span>
                        </div>
                        <div className="flex">
                            <span className="text-[#605e5c] w-20 shrink-0">reply-to:</span>
                            <span className="text-[#252423]">{email?.sender_email}</span>
                        </div>
                        <div className="flex">
                            <span className="text-[#605e5c] w-20 shrink-0">to:</span>
                            <span className="text-[#252423]">me</span>
                        </div>
                        <div className="flex">
                            <span className="text-[#605e5c] w-20 shrink-0">date:</span>
                            <span className="text-[#252423]">
                                {email?.timestamp ? new Date(email.timestamp).toLocaleString() : 'Today'}
                            </span>
                        </div>
                        <div className="flex">
                            <span className="text-[#605e5c] w-20 shrink-0">subject:</span>
                            <span className="text-[#252423]">{email?.subject}</span>
                        </div>

                        <div className="border-t border-gray-100 my-2 pt-2">
                            <div className="flex">
                                <span className="text-[#605e5c] w-20 shrink-0">mailed-by:</span>
                                <span className="text-[#252423]">{domain}</span>
                            </div>
                            <div className="flex">
                                <span className="text-[#605e5c] w-20 shrink-0">signed-by:</span>
                                <span className="text-[#252423]">{domain}</span>
                            </div>
                            <div className="flex items-center">
                                <span className="text-[#605e5c] w-20 shrink-0">security:</span>
                                <span className="flex items-center text-[#252423]">
                                    <Lock size={12} className="mr-1 text-green-600" />
                                    Standard encryption (TLS)
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Close hint */}
                    <div className="mt-3 pt-2 border-t border-gray-100 text-xs text-[#605e5c] text-center">
                        Click arrow to close
                    </div>
                </div>
            )}
        </div>
    );
};

// =========================================================================
// MAIN READING PANE COMPONENT
// =========================================================================

const ReadingPane = ({
    email,
    onAction,
    isLatest,
    actionsTaken,
    onDone,
    isFinished,
    participantId
}) => {
    // Link tooltip state
    const [hoveredLink, setHoveredLink] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

    // Sender details expansion state
    const [senderDetailsExpanded, setSenderDetailsExpanded] = useState(false);

    // Hover tracking refs - with debouncing to prevent double recording
    const hoverStartTime = useRef(null);
    const currentHoverType = useRef(null);
    const currentHoverTarget = useRef(null);
    const lastHoverRecorded = useRef({ type: null, target: null, time: 0 });

    // Session tracking refs
    const sessionStartTime = useRef(null);
    const emailIdRef = useRef(null);

    // Track if link was clicked (separate from final action)
    const linkWasClicked = useRef(false);

    // =========================================================================
    // SESSION MANAGEMENT (session opening handled by EmailApp.jsx)
    // FIX #2: Removed duplicate openEmailSession call
    // =========================================================================

    useEffect(() => {
        if (email && email.id && participantId && isLatest) {
            const currentEmailId = email.id;

            // Only reset tracking if it's a new email
            if (emailIdRef.current !== currentEmailId) {
                emailIdRef.current = currentEmailId;
                sessionStartTime.current = Date.now();
                linkWasClicked.current = false;
                setSenderDetailsExpanded(false); // Reset sender panel
                // NOTE: Session opening is handled by EmailApp.jsx to prevent duplicates
            }
        }

        return () => {
            // Cleanup
        };
    }, [email?.id, participantId, isLatest]);

    // NOTE: openEmailSession function removed - handled by EmailApp.jsx with deduplication

    // =========================================================================
    // MICRO-ACTION TRACKING (with debouncing)
    // =========================================================================

    const recordMicroAction = useCallback(async (actionType, details = {}) => {
        if (!participantId || !email?.id) return;

        // Debounce: Don't record if same action was recorded in last 500ms
        const now = Date.now();
        const lastRecord = lastHoverRecorded.current;

        if (
            lastRecord.type === actionType &&
            lastRecord.target === details.link &&
            (now - lastRecord.time) < 500
        ) {
            return; // Skip duplicate
        }

        // Update last recorded
        lastHoverRecorded.current = {
            type: actionType,
            target: details.link || null,
            time: now
        };

        try {
            await fetch(`${API_BASE}/session/micro-action/${participantId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email_id: email.id,
                    action_type: actionType,
                    details
                })
            });
        } catch (err) {
            console.error('Failed to record micro-action:', err);
        }
    }, [participantId, email?.id]);

    // =========================================================================
    // LINK INTERACTION HANDLERS
    // =========================================================================

    const handleLinkHover = (e, type) => {
        if (e.target.tagName === 'A') {
            const link = e.target.href;

            if (type === 'enter') {
                // Only start tracking if not already tracking this link
                if (currentHoverType.current !== 'link' || currentHoverTarget.current !== link) {
                    hoverStartTime.current = Date.now();
                    currentHoverType.current = 'link';
                    currentHoverTarget.current = link;
                }
                setHoveredLink(link);
                setTooltipPos({ x: e.clientX, y: e.clientY + 20 });

            } else if (type === 'leave') {
                if (hoverStartTime.current && currentHoverType.current === 'link') {
                    const duration = Date.now() - hoverStartTime.current;

                    // Only log if duration is significant (> 100ms)
                    if (duration > 100) {
                        // Record to backend (debounced)
                        recordMicroAction('link_hover', {
                            link: currentHoverTarget.current,
                            duration_ms: duration
                        });

                        // Notify parent handler for UI state
                        onAction('link_hover', { link: currentHoverTarget.current, duration });
                    }
                }

                hoverStartTime.current = null;
                currentHoverType.current = null;
                currentHoverTarget.current = null;
                setHoveredLink(null);
            }
        }
    };

    // =========================================================================
    // FIX #3: Clear hover state before recording link_click
    // =========================================================================

    const handleLinkClick = (e, link) => {
        e.preventDefault();

        // FIX: Clear hover tracking BEFORE recording click to prevent link_hover event
        hoverStartTime.current = null;
        currentHoverType.current = null;
        currentHoverTarget.current = null;
        setHoveredLink(null);

        // Mark that a link was clicked
        linkWasClicked.current = true;

        // Record to backend as micro-action (silent - no visual feedback)
        recordMicroAction('link_click', { link });

        // Notify parent (for session tracking, bonus calculation happens silently)
        onAction('link_click', { link });

        // No fake page overlay - link click is tracked but nothing visible happens
        // This maintains ecological validity - user must judge on their own
    };

    // =========================================================================
    // SENDER CLICK HANDLER (replaces sender_hover)
    // =========================================================================

    const handleSenderClick = () => {
        if (!isLatest) return;

        // Record sender inspection as a click action
        recordMicroAction('sender_click', {
            timestamp: Date.now(),
            email_id: email?.id
        });

        // Notify parent
        onAction('sender_click', { inspected: true });
    };

    // =========================================================================
    // TOOLTIP POSITIONING
    // =========================================================================

    const handleMouseMove = (e) => {
        if (hoveredLink) {
            setTooltipPos({ x: e.clientX, y: e.clientY + 20 });
        }
    };

    // =========================================================================
    // FINAL ACTION HANDLERS
    // =========================================================================

    const handleFinalAction = (actionType) => {
        if (!isLatest) return;

        const responseTime = sessionStartTime.current
            ? Date.now() - sessionStartTime.current
            : 0;

        onAction(actionType, {
            responseTime,
            linkWasClicked: linkWasClicked.current,
            sessionStart: sessionStartTime.current
        });
    };

    // =========================================================================
    // EMAIL BODY FORMATTING
    // =========================================================================

    const formatEmailBody = (body) => {
        if (!body) return '';

        // Add CSS for proper email formatting
        const styledBody = `
            <style>
                .email-content p { margin-bottom: 1em; }
                .email-content p:first-child { margin-top: 0; }
                .email-content p:last-child { margin-bottom: 0; }
                .email-content ul, .email-content ol { margin: 1em 0; padding-left: 1.5em; }
                .email-content li { margin-bottom: 0.5em; }
                .email-content a { 
                    color: #0066cc !important; 
                    text-decoration: underline !important;
                    background-color: #e6f3ff;
                    padding: 2px 4px;
                    border-radius: 2px;
                    font-weight: 500;
                }
                .email-content a:hover { 
                    background-color: #cce5ff;
                    color: #004499 !important;
                }
                .email-content br + br { display: block; margin: 0.5em 0; }
            </style>
            <div class="email-content">${body}</div>
        `;

        return styledBody;
    };

    // =========================================================================
    // RENDER: FINISHED STATE (show message when no email selected, show email when selected)
    // =========================================================================

    if (isFinished && !email) {
        return (
            <div className="flex-1 bg-white flex items-center justify-center">
                <div className="text-center">
                    <h2 className="text-2xl font-semibold text-[#252423] mb-2">That's all.</h2>
                    <p className="text-[#605e5c] text-lg">Thanks for your time.</p>
                    <p className="text-sm text-[#a19f9d] mt-4">Click any email on the left to review it.</p>
                </div>
            </div>
        );
    }

    // =========================================================================
    // RENDER: NO EMAIL SELECTED (during study)
    // =========================================================================

    if (!email) {
        return (
            <div className="flex-1 bg-[#f3f2f1] flex items-center justify-center text-[#605e5c]">
                <div className="text-center">
                    <div className="text-lg font-semibold mb-1">No email selected</div>
                    <div className="text-sm">Select an email to view</div>
                </div>
            </div>
        );
    }

    // =========================================================================
    // RENDER: WELCOME EMAIL (order_id === 0)
    // =========================================================================

    if (email.order_id === 0) {
        return (
            <div className="flex-1 bg-white flex flex-col overflow-hidden">
                {/* Email Header */}
                <div className="p-6 border-b border-gray-200 bg-white">
                    {/* Subject line */}
                    <div className="flex justify-between items-start mb-6">
                        <h1 className="text-xl font-semibold text-[#252423] leading-tight flex-1 mr-4">
                            {email.subject}
                        </h1>
                    </div>
                    <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-4">
                            <div className="w-12 h-12 rounded-full bg-[#0078d4] flex items-center justify-center text-white font-bold text-lg">
                                R
                            </div>
                            <div>
                                <div className="font-semibold text-lg text-[#252423]">
                                    {email.sender_name}
                                </div>
                                <div className="text-xs text-[#605e5c] mt-0.5">
                                    to me
                                </div>
                            </div>
                        </div>
                        <div className="text-xs text-[#605e5c]">
                            <span>
                                {new Date(email.timestamp).toLocaleString([], {
                                    weekday: 'short',
                                    year: 'numeric',
                                    month: 'numeric',
                                    day: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Email Body */}
                <div className="flex-1 overflow-y-auto p-6">
                    <div
                        className="prose max-w-none text-[#252423] leading-relaxed"
                        dangerouslySetInnerHTML={{ __html: formatEmailBody(email.body) }}
                    />
                </div>

                {/* Auto-advance notice */}
                <div className="p-4 bg-blue-50 border-t border-blue-100 text-center">
                    <p className="text-sm text-blue-700">
                        Please read this welcome message carefully. The first email will arrive shortly...
                    </p>
                </div>
            </div>
        );
    }

    // =========================================================================
    // RENDER: REGULAR EMAIL
    // =========================================================================

    return (
        <div className="flex-1 bg-white flex flex-col overflow-hidden" onMouseMove={handleMouseMove}>
            {/* Email Header with Evaluation Done badge */}
            <div className="p-6 border-b border-gray-200 bg-white">
                {/* Subject line */}
                <div className="flex justify-between items-start mb-6">
                    <h1 className="text-xl font-semibold text-[#252423] leading-tight flex-1 mr-4">
                        {email.subject}
                    </h1>
                    {/* NO "Link clicked" indicator - removed for ecological validity */}
                </div>

                <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 rounded-full bg-[#c7e0f4] flex items-center justify-center text-[#0078d4] font-bold text-lg">
                            {email.sender_name.charAt(0)}
                        </div>
                        <div>
                            {/* Sender Name ONLY - no email address visible */}
                            <div className="font-semibold text-lg text-[#252423]">
                                {email.sender_name}
                            </div>
                            {/* Expandable "to me" with sender details */}
                            <SenderDetailsPanel
                                email={email}
                                isExpanded={senderDetailsExpanded}
                                onToggle={() => setSenderDetailsExpanded(!senderDetailsExpanded)}
                                onSenderClick={handleSenderClick}
                            />
                        </div>
                    </div>

                    {/* Right side: Time OR Evaluation Done badge */}
                    <div className="flex items-center space-x-3">
                        {actionsTaken && isLatest && email.order_id !== 0 ? (
                            <div className="flex items-center space-x-2 bg-green-50 text-green-700 px-4 py-2 rounded-full animate-pulse">
                                <CheckCircle size={18} strokeWidth={2} />
                                <span className="font-semibold text-sm">Evaluation Done</span>
                            </div>
                        ) : (
                            <div className="text-xs text-[#605e5c]">
                                <span>
                                    {new Date(email.timestamp).toLocaleString([], {
                                        weekday: 'short',
                                        year: 'numeric',
                                        month: 'numeric',
                                        day: 'numeric',
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    })}
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Toolbar */}
            <div className="flex items-center justify-between px-6 py-2 border-b border-gray-100 bg-[#faf9f8]">
                <div className="flex items-center space-x-1">
                    <button className="flex items-center space-x-1.5 px-3 py-1.5 text-sm text-[#252423] hover:bg-[#edebe9] rounded transition-colors">
                        <Reply size={16} strokeWidth={1.5} />
                        <span>Reply</span>
                    </button>
                    <button className="flex items-center space-x-1.5 px-3 py-1.5 text-sm text-[#252423] hover:bg-[#edebe9] rounded transition-colors">
                        <ReplyAll size={16} strokeWidth={1.5} />
                        <span>Reply all</span>
                    </button>
                    <button className="flex items-center space-x-1.5 px-3 py-1.5 text-sm text-[#252423] hover:bg-[#edebe9] rounded transition-colors">
                        <Forward size={16} strokeWidth={1.5} />
                        <span>Forward</span>
                    </button>
                    <div className="w-px h-5 bg-gray-300 mx-2"></div>
                    <button className="p-1.5 text-[#252423] hover:bg-[#edebe9] rounded transition-colors">
                        <MoreHorizontal size={18} strokeWidth={1.5} />
                    </button>
                </div>
            </div>

            {/* Email Body */}
            <div
                className="flex-1 overflow-y-auto p-6"
                onMouseOver={(e) => handleLinkHover(e, 'enter')}
                onMouseOut={(e) => handleLinkHover(e, 'leave')}
                onClick={(e) => {
                    if (e.target.tagName === 'A') {
                        handleLinkClick(e, e.target.href);
                    }
                }}
            >
                <div
                    className="prose max-w-none text-[#252423] leading-relaxed"
                    style={{ lineHeight: '1.6' }}
                    dangerouslySetInnerHTML={{ __html: formatEmailBody(email.body) }}
                />
            </div>

            {/* Link Tooltip */}
            {hoveredLink && (
                <div
                    className="fixed bg-gray-800 text-white text-xs px-2 py-1 rounded shadow-lg z-50 max-w-md truncate"
                    style={{ left: tooltipPos.x, top: tooltipPos.y }}
                >
                    {hoveredLink}
                </div>
            )}

            {/* Action Buttons - Only for latest email when not finished */}
            {isLatest && !actionsTaken && !isFinished && (
                <div className="p-4 border-t border-gray-200 bg-[#faf9f8]">
                    <div className="flex items-center justify-center space-x-3">
                        <button
                            onClick={() => handleFinalAction('safe')}
                            className="flex items-center space-x-2 px-5 py-2.5 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors font-medium shadow-sm"
                        >
                            <CheckCircle size={18} />
                            <span>Mark Safe</span>
                        </button>
                        <button
                            onClick={() => handleFinalAction('report')}
                            className="flex items-center space-x-2 px-5 py-2.5 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors font-medium shadow-sm"
                        >
                            <AlertOctagon size={18} />
                            <span>Report Phishing</span>
                        </button>
                        <button
                            onClick={() => handleFinalAction('delete')}
                            className="flex items-center space-x-2 px-5 py-2.5 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors font-medium shadow-sm"
                        >
                            <Trash2 size={18} />
                            <span>Delete</span>
                        </button>
                        <button
                            onClick={() => handleFinalAction('ignore')}
                            className="flex items-center space-x-2 px-5 py-2.5 bg-gray-400 text-white rounded-md hover:bg-gray-500 transition-colors font-medium shadow-sm"
                        >
                            <Ban size={18} />
                            <span>Ignore</span>
                        </button>
                    </div>
                </div>
            )}

            {/* Continue Button - Shows after action taken (only when not finished) */}
            {isLatest && actionsTaken && !isFinished && (
                <div className="p-4 border-t border-gray-200 bg-green-50">
                    <div className="flex items-center justify-center">
                        <button
                            onClick={onDone}
                            className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-semibold shadow-md"
                        >
                            <CheckCircle size={20} />
                            <span>Continue to Next Email</span>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ReadingPane;