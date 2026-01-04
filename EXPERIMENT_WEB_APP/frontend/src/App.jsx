/**
 * CYPEARL Experiment Web App - Main Application (UPDATED v3 - FIXED)
 * 
 * FIXES IN THIS VERSION:
 * 1. Issue #4: Delete action now auto-advances to next email
 * 2. Issue #5: Users can view emails after study completion (read-only)
 * 
 * FEATURES:
 * - No bonus display in header - bonus is tracked silently
 * - Bonus calculation still happens in backend (revealed at end)
 * - Fixed duplicate email_open and mark_read events with refs
 * - sender_hover replaced with sender_click tracking
 * - Link clicks tracked silently without visible feedback
 * 
 * Tracks all observational data required for phishing_study_responses.csv:
 * - response_latency_ms: Time from email open to action
 * - dwell_time_ms: Total time viewing email  
 * - clicked: Whether any link was clicked (separate from action)
 * - hovered_link: Whether links were hovered
 * - inspected_sender: Whether user clicked to expand sender details
 * - confidence_rating: Self-reported confidence (1-10)
 * - suspicion_rating: Self-reported suspicion (1-10)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import EmailList from './components/EmailList';
import ReadingPane from './components/ReadingPane';
import ActionModal from './components/ActionModal';
import { Search, Check, Settings, HelpCircle, Mail, Calendar, Users, Paperclip, Trash2 } from 'lucide-react';
import axios from 'axios';

// Get participant ID from localStorage
const getParticipantId = () => {
    let id = localStorage.getItem('participant_id');
    return id;
};

const getClientInfo = () => ({
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    window_width: window.innerWidth,
    window_height: window.innerHeight,
    pixel_ratio: window.devicePixelRatio,
    user_agent: navigator.userAgent
});

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// =========================================================================
// MAIN APP COMPONENT
// =========================================================================

function App() {
    // Auth state
    const [participantId, setParticipantId] = useState(getParticipantId());
    const [showIdPrompt, setShowIdPrompt] = useState(!getParticipantId());
    const [prolificIdInput, setProlificIdInput] = useState('');

    // Email state
    const [emailList, setEmailList] = useState([]);
    const [selectedEmailId, setSelectedEmailId] = useState(null);
    const [loading, setLoading] = useState(true);
    const [finished, setFinished] = useState(false);
    const [activeFolder, setActiveFolder] = useState('inbox');

    // UI state
    const [modalOpen, setModalOpen] = useState(false);
    const [pendingAction, setPendingAction] = useState(null);
    const [actionsTaken, setActionsTaken] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);

    // Counts
    const [unreadCount, setUnreadCount] = useState(0);
    const [deletedCount, setDeletedCount] = useState(0);

    // Session tracking for observational data
    const [emailOpenTime, setEmailOpenTime] = useState(null);
    const [currentSessionData, setCurrentSessionData] = useState({
        linkClicked: false,
        linkHovered: false,
        senderInspected: false,  // Now tracks sender_click
        linkHoverCount: 0,
        senderClickCount: 0
    });

    // Refs to prevent duplicate API calls
    const openedEmailsRef = useRef(new Set()); // Track emails we've opened sessions for
    const markedReadRef = useRef(new Set());   // Track emails we've marked as read

    // =========================================================================
    // EMAIL FETCHING
    // =========================================================================

    const fetchEmails = async (folder = activeFolder, targetSelectionId = null) => {
        if (!participantId) return;

        setLoading(true);
        try {
            const response = await axios.get(`${API_URL}/emails/inbox/${participantId}?folder=${folder}`);
            const data = response.data;
            const emails = data.emails || [];

            if (data.counts) {
                setUnreadCount(data.counts.unread);
                setDeletedCount(data.counts.deleted);
            }

            setEmailList(emails);

            // Selection Logic
            if (targetSelectionId && emails.find(e => e.id === targetSelectionId)) {
                setSelectedEmailId(targetSelectionId);
            } else if (!selectedEmailId && emails.length > 0) {
                setSelectedEmailId(emails[0].id);
            } else if (selectedEmailId && !emails.find(e => e.id === selectedEmailId)) {
                setSelectedEmailId(emails.length > 0 ? emails[0].id : null);
            }

            // Check for finished state
            if (data.is_finished !== undefined) {
                setFinished(data.is_finished);
            } else {
                if (folder === 'inbox' && emails.length === 0) {
                    setFinished(true);
                } else {
                    setFinished(false);
                }
            }

        } catch (error) {
            if (error.response && error.response.status === 404) {
                console.log("Participant not found, resetting session");
                localStorage.removeItem('participant_id');
                setParticipantId(null);
                setShowIdPrompt(true);
                setEmailList([]);
                setFinished(false);
            } else {
                console.error("Error fetching emails:", error);
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (participantId) {
            fetchEmails(activeFolder);
        }
    }, [activeFolder, participantId]);

    // =========================================================================
    // CLEAR SELECTION WHEN STUDY FINISHES
    // =========================================================================

    useEffect(() => {
        if (finished) {
            // Clear selection so "That's all" message shows first
            setSelectedEmailId(null);
        }
    }, [finished]);

    // =========================================================================
    // AUTO-ADVANCE FOR WELCOME EMAIL
    // =========================================================================

    useEffect(() => {
        if (emailList.length > 0 && emailList[0].order_id === 0 && !finished) {
            const timer = setTimeout(async () => {
                try {
                    await axios.post(`${API_URL}/complete/${participantId}`);
                    const response = await axios.get(`${API_URL}/emails/inbox/${participantId}?folder=${activeFolder}`);
                    const data = response.data;
                    const emails = data.emails || [];

                    if (data.counts) {
                        setUnreadCount(data.counts.unread);
                        setDeletedCount(data.counts.deleted);
                    }

                    setEmailList(emails);

                    if (data.is_finished !== undefined) {
                        setFinished(data.is_finished);
                    } else if (activeFolder === 'inbox' && emails.length === 0) {
                        setFinished(true);
                    }
                } catch (error) {
                    console.error("Error auto-advancing:", error);
                }
            }, 8000); // Give more time to read welcome email
            return () => clearTimeout(timer);
        }
    }, [emailList, finished]);

    // =========================================================================
    // SESSION TRACKING - OPEN EMAIL SESSION (prevents duplicate opens)
    // =========================================================================

    useEffect(() => {
        if (selectedEmailId && participantId && emailList.length > 0) {
            const isLatestEmail = selectedEmailId === emailList[0].id;

            if (isLatestEmail && activeFolder === 'inbox' && !finished) {
                // Check if we've already opened this email session
                if (!openedEmailsRef.current.has(selectedEmailId)) {
                    // Start tracking session
                    setEmailOpenTime(Date.now());
                    setActionsTaken(false);

                    // Reset session tracking data
                    setCurrentSessionData({
                        linkClicked: false,
                        linkHovered: false,
                        senderInspected: false,
                        linkHoverCount: 0,
                        senderClickCount: 0
                    });

                    // Mark this email as opened
                    openedEmailsRef.current.add(selectedEmailId);

                    // Notify backend of session open (only once)
                    openEmailSession(selectedEmailId);
                }
            }
        }
    }, [selectedEmailId, emailList, participantId, activeFolder, finished]);

    const openEmailSession = async (emailId) => {
        try {
            await axios.post(`${API_URL}/session/open/${participantId}`, {
                email_id: emailId
            });
        } catch (error) {
            console.log("Session open endpoint not available, using fallback");
        }
    };

    // =========================================================================
    // MARK EMAIL AS READ (prevents duplicate mark_read)
    // =========================================================================

    useEffect(() => {
        if (selectedEmailId && activeFolder === 'inbox' && !finished) {
            const email = emailList.find(e => e.id === selectedEmailId);

            // Only mark as read if not already marked
            if (email && !email.is_read && !markedReadRef.current.has(selectedEmailId)) {
                markedReadRef.current.add(selectedEmailId);

                axios.post(`${API_URL}/actions/${participantId}`, {
                    email_id: selectedEmailId,
                    action_type: 'mark_read'
                }).then(() => {
                    setEmailList(prev => prev.map(e =>
                        e.id === selectedEmailId ? { ...e, is_read: true } : e
                    ));
                    setUnreadCount(prev => Math.max(0, prev - 1));
                }).catch(err => {
                    // Remove from ref if failed so it can retry
                    markedReadRef.current.delete(selectedEmailId);
                });
            }
        }
    }, [selectedEmailId, finished]);

    // =========================================================================
    // ACTION HANDLING
    // =========================================================================

    const handleAction = async (actionType, data = {}) => {
        // Handle micro-actions (tracking events)
        if (['link_hover', 'link_click', 'sender_click'].includes(actionType)) {
            // Update local session tracking
            setCurrentSessionData(prev => {
                const updated = { ...prev };

                if (actionType === 'link_hover') {
                    updated.linkHovered = true;
                    updated.linkHoverCount = (prev.linkHoverCount || 0) + 1;
                } else if (actionType === 'link_click') {
                    updated.linkClicked = true;

                    // Silent bonus calculation - send to backend but don't display
                    calculateBonusSilently(data.link);
                } else if (actionType === 'sender_click') {
                    updated.senderInspected = true;
                    updated.senderClickCount = (prev.senderClickCount || 0) + 1;
                }

                return updated;
            });

            return;
        }

        // Handle final actions (safe, report, delete, ignore)
        if (['safe', 'report', 'delete', 'ignore'].includes(actionType)) {
            setPendingAction(actionType);
            setModalOpen(true);
        }
    };

    // =========================================================================
    // SILENT BONUS CALCULATION (no UI feedback)
    // =========================================================================

    const calculateBonusSilently = async (link) => {
        if (!participantId || !link) return;

        try {
            // Backend will calculate and store bonus, but we don't display it
            await axios.post(`${API_URL}/bonus/calculate/${participantId}`, {
                link: link,
                email_id: selectedEmailId,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            // Silently fail - this is just bonus tracking
            console.log("Bonus calculation sent to backend");
        }
    };

    // =========================================================================
    // SUBMIT FINAL ACTION
    // FIX #4: Delete action now auto-advances to next email
    // =========================================================================

    const handleSubmitAction = async ({ reason, confidence, suspicion }) => {
        if (!selectedEmailId || !pendingAction) return;

        const dwellTime = emailOpenTime ? Date.now() - emailOpenTime : 0;
        const latency = emailOpenTime ? Date.now() - emailOpenTime : 0;
        const isDeleteAction = pendingAction === 'delete';

        try {
            await axios.post(`${API_URL}/actions/${participantId}`, {
                email_id: selectedEmailId,
                action_type: pendingAction,
                reason: reason,
                confidence: confidence,
                suspicion: suspicion,
                latency_ms: latency,
                dwell_time_ms: dwellTime,
                clicked_link: currentSessionData.linkClicked,
                hovered_link: currentSessionData.linkHovered,
                inspected_sender: currentSessionData.senderInspected,
                link_hover_count: currentSessionData.linkHoverCount,
                sender_click_count: currentSessionData.senderClickCount,
                client_info: getClientInfo()
            });

            setModalOpen(false);
            setPendingAction(null);

            // FIX #4 (UPDATED): Handle delete - select previous email, new email appears unread
            if (isDeleteAction) {
                setDeletedCount(prev => prev + 1);

                // Find the index of the deleted email in the list
                // emailList is sorted by order_id DESC (newest first)
                const deletedIndex = emailList.findIndex(e => e.id === selectedEmailId);

                // Get the email to select after delete (the one below in the list = older email)
                const nextIndex = deletedIndex < emailList.length - 1
                    ? deletedIndex + 1
                    : Math.max(0, deletedIndex - 1);
                const emailToSelectAfterDelete = emailList[nextIndex];
                const targetEmailId = emailToSelectAfterDelete?.id;

                // Remove the deleted email from local list immediately
                setEmailList(prev => prev.filter(e => e.id !== selectedEmailId));

                // Advance to next email order (so new email appears at top as unread)
                try {
                    await axios.post(`${API_URL}/complete/${participantId}`);

                    // Clear tracking refs for the new email
                    openedEmailsRef.current.clear();
                    markedReadRef.current.clear();

                    // Fetch updated emails, selecting the PREVIOUS email (not the new one)
                    await fetchEmails(activeFolder, targetEmailId);

                    // Reset session state
                    setActionsTaken(false);
                    setEmailOpenTime(null);
                    setCurrentSessionData({
                        linkClicked: false,
                        linkHovered: false,
                        senderInspected: false,
                        linkHoverCount: 0,
                        senderClickCount: 0
                    });
                } catch (error) {
                    console.error("Error after delete:", error);
                }

                return; // Don't set actionsTaken
            }

            // For non-delete actions, show "Continue" button
            setActionsTaken(true);

        } catch (error) {
            console.error("Error submitting action:", error);
            alert("Failed to submit action. Please try again.");
        }
    };

    // =========================================================================
    // NAVIGATION
    // =========================================================================

    const handleDone = async () => {
        if (!selectedEmailId && !finished) return;

        setIsTransitioning(true);

        try {
            // Advance to next email
            await axios.post(`${API_URL}/complete/${participantId}`);

            // Clear refs for new email
            openedEmailsRef.current.clear();
            markedReadRef.current.clear();

            // Fetch updated emails
            await fetchEmails(activeFolder);

            // Reset session state
            setActionsTaken(false);
            setEmailOpenTime(null);
            setCurrentSessionData({
                linkClicked: false,
                linkHovered: false,
                senderInspected: false,
                linkHoverCount: 0,
                senderClickCount: 0
            });

        } catch (error) {
            console.error("Error advancing to next email:", error);
        } finally {
            setIsTransitioning(false);
        }
    };

    // =========================================================================
    // AUTH HANDLERS
    // =========================================================================

    const handleProlificSubmit = async (e) => {
        e.preventDefault();
        if (!prolificIdInput.trim()) return;

        try {
            const response = await axios.post(`${API_URL}/auth/login`, {
                prolific_id: prolificIdInput.trim(),
                user_agent: navigator.userAgent,
                screen_resolution: `${window.screen.width}x${window.screen.height}`
            });

            const newParticipantId = response.data.participant_id;
            localStorage.setItem('participant_id', newParticipantId);
            setParticipantId(newParticipantId);
            setShowIdPrompt(false);

        } catch (error) {
            console.error("Login error:", error);
            alert("Failed to start session. Please try again.");
        }
    };

    const handleReset = async () => {
        if (confirm("Reset your session? This will clear all progress.")) {
            localStorage.removeItem('participant_id');
            openedEmailsRef.current.clear();
            markedReadRef.current.clear();
            setParticipantId(null);
            setShowIdPrompt(true);
            setEmailList([]);
            setFinished(false);
            setSelectedEmailId(null);
            setActionsTaken(false);
        }
    };

    // =========================================================================
    // DERIVED STATE
    // =========================================================================

    const selectedEmail = emailList.find(e => e.id === selectedEmailId);
    const isLatest = emailList.length > 0 && selectedEmailId === emailList[0].id;

    // =========================================================================
    // RENDER: LOGIN PROMPT
    // =========================================================================

    if (showIdPrompt) {
        return (
            <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
                <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
                    <h1 className="text-2xl font-bold mb-6 text-center text-[#0078d4]">Welcome</h1>
                    <form onSubmit={handleProlificSubmit} className="space-y-4">
                        <div>
                            <label htmlFor="prolificId" className="block text-sm font-medium text-gray-700 mb-1">
                                Please enter your Prolific ID to start with...
                            </label>
                            <input
                                type="text"
                                id="prolificId"
                                value={prolificIdInput}
                                onChange={(e) => setProlificIdInput(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Enter ID here"
                                required
                            />
                        </div>
                        <button
                            type="submit"
                            className="w-full bg-[#0078d4] text-white py-2 px-4 rounded-md hover:bg-[#106ebe] transition-colors font-semibold"
                        >
                            Start Evaluation
                        </button>
                    </form>
                </div>
            </div>
        );
    }

    // =========================================================================
    // RENDER: LOADING
    // =========================================================================

    if (loading && emailList.length === 0 && !finished) {
        return (
            <div className="flex h-screen items-center justify-center bg-gray-100">
                <div className="flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <div className="text-blue-600 font-semibold">Loading ProMail Suite...</div>
                </div>
            </div>
        );
    }

    // =========================================================================
    // RENDER: MAIN APP
    // =========================================================================

    return (
        <div className="flex h-screen bg-white overflow-hidden font-sans">
            {/* OWA Header - NO bonus display */}
            <div className="absolute top-0 left-0 w-full h-[48px] bg-[#0078d4] flex items-center justify-between px-2 z-50 text-white select-none">
                <div className="flex items-center">
                    <div className="p-3 hover:bg-[#106ebe] cursor-pointer transition-colors rounded-sm mr-1">
                        <div className="grid grid-cols-3 gap-[2px]">
                            {[...Array(9)].map((_, i) => (
                                <div key={i} className="w-[3px] h-[3px] bg-white rounded-full"></div>
                            ))}
                        </div>
                    </div>
                    <span className="font-semibold text-base tracking-wide ml-1">ProMail Suite</span>
                </div>

                {/* Search Bar */}
                <div className="flex-1 max-w-2xl mx-4">
                    <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Search size={16} className="text-[#0078d4]" />
                        </div>
                        <input
                            type="text"
                            placeholder="Search"
                            className="block w-full pl-10 pr-3 py-1.5 border-none rounded-[4px] leading-5 bg-[#c3e0fa] text-black placeholder-gray-600 focus:outline-none focus:bg-white focus:ring-0 transition-colors h-[32px] text-sm"
                        />
                    </div>
                </div>

                {/* Right Actions */}
                <div className="flex items-center space-x-1">
                    <div className="w-8 h-8 flex items-center justify-center hover:bg-[#106ebe] rounded-sm cursor-pointer" onClick={handleReset} title="Reset Simulation">
                        <Settings size={20} strokeWidth={1.5} />
                    </div>
                    <div className="w-8 h-8 flex items-center justify-center hover:bg-[#106ebe] rounded-sm cursor-pointer">
                        <HelpCircle size={20} strokeWidth={1.5} />
                    </div>
                    <div className="ml-2 w-8 h-8 rounded-full bg-[#004578] flex items-center justify-center text-xs font-bold border-2 border-white/20 cursor-pointer hover:opacity-90">
                        JD
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex w-full h-full pt-[48px]">
                {/* Left Rail */}
                <div className="w-[48px] bg-[#f3f2f1] flex flex-col items-center py-2 space-y-4 border-r border-gray-200 z-40">
                    <div className="p-2 text-[#0078d4] bg-white shadow-sm rounded cursor-pointer">
                        <Mail size={20} strokeWidth={1.5} />
                    </div>
                    <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
                        <Calendar size={20} strokeWidth={1.5} />
                    </div>
                    <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
                        <Users size={20} strokeWidth={1.5} />
                    </div>
                    <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
                        <Paperclip size={20} strokeWidth={1.5} />
                    </div>
                </div>

                <Sidebar
                    activeFolder={activeFolder}
                    onFolderSelect={setActiveFolder}
                    unreadCount={unreadCount}
                    deletedCount={deletedCount}
                />

                {/* FIX #5: Allow email viewing after completion */}
                {finished ? (
                    <>
                        <EmailList
                            emails={emailList}
                            selectedId={selectedEmailId}
                            onSelect={setSelectedEmailId}
                            onDone={() => { }}
                            actionsTaken={true}
                        />
                        <ReadingPane
                            email={selectedEmail}
                            onAction={() => { }}
                            isLatest={false}
                            actionsTaken={true}
                            onDone={() => { }}
                            isFinished={true}
                            participantId={participantId}
                        />
                    </>
                ) : (activeFolder === 'inbox' || activeFolder === 'deleted') ? (
                    <>
                        <EmailList
                            emails={emailList}
                            selectedId={selectedEmailId}
                            onSelect={setSelectedEmailId}
                            onDone={handleDone}
                            actionsTaken={actionsTaken}
                        />
                        <ReadingPane
                            email={selectedEmail}
                            onAction={handleAction}
                            isLatest={isLatest && !isTransitioning}
                            actionsTaken={actionsTaken}
                            onDone={handleDone}
                            isFinished={false}
                            participantId={participantId}
                        />
                    </>
                ) : (
                    <div className="flex-1 flex items-center justify-center bg-[#f3f2f1] text-[#605e5c]">
                        <div className="text-center">
                            <div className="text-lg font-semibold mb-1">Nothing here</div>
                            <div className="text-sm">This folder is empty</div>
                        </div>
                    </div>
                )}
            </div>

            <ActionModal
                isOpen={modalOpen}
                onClose={() => setModalOpen(false)}
                onSubmit={handleSubmitAction}
                actionType={pendingAction}
            />
        </div>
    );
}

export default App;