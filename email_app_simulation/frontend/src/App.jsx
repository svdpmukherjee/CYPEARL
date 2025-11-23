import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import EmailList from './components/EmailList';
import ReadingPane from './components/ReadingPane';
import ActionModal from './components/ActionModal';
import { Search, Check, Settings, HelpCircle, Mail, Calendar, Users, Paperclip, Trash2 } from 'lucide-react';
import axios from 'axios';

// Generate a random participant ID for this session if not exists
const getParticipantId = () => {
    let id = localStorage.getItem('participant_id');
    // REMOVED auto-generation to force prompt
    // if (!id) {
    //     id = 'user_' + Math.random().toString(36).substr(2, 9);
    //     localStorage.setItem('participant_id', id);
    // }
    return id;
};

const getClientInfo = () => ({
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    window_width: window.innerWidth,
    window_height: window.innerHeight,
    pixel_ratio: window.devicePixelRatio
});

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
    const [emailList, setEmailList] = useState([]);
    const [selectedEmailId, setSelectedEmailId] = useState(null);
    const [loading, setLoading] = useState(true);
    const [modalOpen, setModalOpen] = useState(false);
    const [pendingAction, setPendingAction] = useState(null);
    const [participantId, setParticipantId] = useState(getParticipantId());
    const [finished, setFinished] = useState(false);
    const [activeFolder, setActiveFolder] = useState('inbox');
    const [actionsTaken, setActionsTaken] = useState(false); // Track if user has taken an action on current email
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [showIdPrompt, setShowIdPrompt] = useState(!getParticipantId());
    const [prolificIdInput, setProlificIdInput] = useState('');

    const [unreadCount, setUnreadCount] = useState(0);
    const [deletedCount, setDeletedCount] = useState(0);
    const [emailOpenTime, setEmailOpenTime] = useState(null);

    const fetchEmails = async (folder = activeFolder, targetSelectionId = null) => {
        if (!participantId) return; // Don't fetch if no ID

        setLoading(true);
        try {
            const response = await axios.get(`${API_URL}/emails/inbox/${participantId}?folder=${folder}`);
            // Handle new response structure
            const data = response.data;
            const emails = data.emails || [];

            if (data.counts) {
                setUnreadCount(data.counts.unread);
                setDeletedCount(data.counts.deleted);
            }

            setEmailList(emails);
            console.log('Fetched emails:', emails.map(e => ({ order_id: e.order_id, subject: e.subject, is_read: e.is_read })));

            // Selection Logic
            if (targetSelectionId && emails.find(e => e.id === targetSelectionId)) {
                // 1. If a specific target is requested and exists, select it
                setSelectedEmailId(targetSelectionId);
            } else if (!selectedEmailId && emails.length > 0) {
                // 2. If nothing selected, select the first one
                setSelectedEmailId(emails[0].id);
            } else if (selectedEmailId && !emails.find(e => e.id === selectedEmailId)) {
                // 3. If current selection is gone (e.g. deleted/moved) and no target specified, select top
                setSelectedEmailId(emails.length > 0 ? emails[0].id : null);
            }

            // Check for finished state
            // Use explicit flag from backend if available, otherwise fallback to empty inbox logic
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
                // Participant not found (e.g. backend reset)
                // Clear local storage and show prompt
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

    // Initial Flow: Auto-advance after 10s if on Email 0
    useEffect(() => {
        if (emailList.length > 0 && emailList[0].order_id === 0 && !finished) {
            const timer = setTimeout(async () => {
                // Advance to next email without auto-selecting it
                try {
                    await axios.post(`${API_URL}/complete/${participantId}`);
                    // Fetch new emails but don't change selection
                    const response = await axios.get(`${API_URL}/emails/inbox/${participantId}?folder=${activeFolder}`);
                    const data = response.data;
                    const emails = data.emails || [];

                    if (data.counts) {
                        setUnreadCount(data.counts.unread);
                        setDeletedCount(data.counts.deleted);
                    }

                    setEmailList(emails);

                    // Check finished state here too
                    if (data.is_finished !== undefined) {
                        setFinished(data.is_finished);
                    } else if (activeFolder === 'inbox' && emails.length === 0) {
                        setFinished(true);
                    }
                } catch (error) {
                    console.error("Error auto-advancing:", error);
                }
            }, 5000); // 5 seconds
            return () => clearTimeout(timer);
        }
    }, [emailList, finished]);

    // Track Open Time when selecting latest email
    useEffect(() => {
        if (selectedEmailId && emailList.length > 0 && selectedEmailId === emailList[0].id) {
            setEmailOpenTime(Date.now());
            setActionsTaken(false);
        }
    }, [selectedEmailId, emailList]);

    // Mark read when selecting an email (only when selection changes)
    useEffect(() => {
        if (selectedEmailId && activeFolder === 'inbox') {
            const email = emailList.find(e => e.id === selectedEmailId);

            console.log('Mark-read effect triggered. selectedEmailId:', selectedEmailId, 'email:', email ? `order_id ${email.order_id}, is_read: ${email.is_read}` : 'not found');

            if (email && !email.is_read) {
                console.log('Marking email as read:', email.order_id, email.subject);
                axios.post(`${API_URL}/actions/${participantId}`, {
                    email_id: selectedEmailId,
                    action_type: 'mark_read'
                }).then(() => {
                    // Update local state to reflect read status
                    setEmailList(prev => prev.map(e =>
                        e.id === selectedEmailId ? { ...e, is_read: true } : e
                    ));
                    // Decrement unread count locally
                    setUnreadCount(prev => Math.max(0, prev - 1));
                });
            }
        }
    }, [selectedEmailId]); // Only depend on selectedEmailId, not emailList

    const handleAction = async (actionType, data = {}) => {
        // Handle immediate actions (link tracking)
        if (['link_hover', 'link_click', 'sender_hover'].includes(actionType)) {
            try {
                await axios.post(`${API_URL}/actions/${participantId}`, {
                    email_id: selectedEmailId,
                    action_type: actionType,
                    hover_data: data, // Wrap the data in hover_data field
                    client_info: getClientInfo()
                });
            } catch (error) {
                console.error("Error submitting tracking:", error);
            }
            return;
        }

        // For decision actions (safe, report, delete, ignore), open modal
        if (['safe', 'report', 'delete', 'ignore'].includes(actionType)) {
            setPendingAction(actionType);
            setModalOpen(true);
        }
    };

    const handleSubmitAction = async ({ reason, confidence }) => {
        try {
            // Calculate latency
            let latency = null;
            if (emailOpenTime) {
                latency = Date.now() - emailOpenTime;
            }

            // Calculate next email to select BEFORE deletion
            let nextEmailId = null;
            if (pendingAction === 'delete') {
                const currentIndex = emailList.findIndex(e => e.id === selectedEmailId);
                if (currentIndex !== -1) {
                    // Try to select the next one (index + 1)
                    if (currentIndex + 1 < emailList.length) {
                        nextEmailId = emailList[currentIndex + 1].id;
                    } else if (currentIndex - 1 >= 0) {
                        // If it was the last one, select the previous one
                        nextEmailId = emailList[currentIndex - 1].id;
                    }
                }
            }

            await axios.post(`${API_URL}/actions/${participantId}`, {
                email_id: selectedEmailId,
                action_type: pendingAction,
                reason,
                confidence,
                latency_ms: latency,
                client_info: getClientInfo()
            });

            setModalOpen(false);
            setPendingAction(null);
            setActionsTaken(true); // Enable "Done" button

            // If deleted, special workflow:
            // 1. Refresh list (moves email to deleted, selects next one)
            // 2. Wait 3 seconds
            // 3. Call complete (triggers next email)
            // 4. Refresh list again
            if (pendingAction === 'delete') {
                setIsTransitioning(true);
                await fetchEmails(activeFolder, nextEmailId);

                // Wait 3 seconds
                await new Promise(resolve => setTimeout(resolve, 3000));

                try {
                    await axios.post(`${API_URL}/complete/${participantId}`);
                    await fetchEmails(activeFolder, nextEmailId); // Fetch new email, keep selection
                } catch (err) {
                    console.error("Error completing after delete:", err);
                } finally {
                    setIsTransitioning(false);
                }
            }

        } catch (error) {
            console.error("Error submitting action:", error);
            alert("Failed to submit action");
            setIsTransitioning(false);
        }
    };

    const handleDone = async () => {
        // Disable button immediately to prevent double clicks and give feedback
        setActionsTaken(false);

        // 3 second delay
        await new Promise(resolve => setTimeout(resolve, 3000));

        try {
            await axios.post(`${API_URL}/complete/${participantId}`);
            // Reset selection to force selecting the new top email
            // setSelectedEmailId(null); // REMOVED to preserve selection
            setActionsTaken(false);
            setEmailOpenTime(null);
            fetchEmails();
        } catch (error) {
            console.error("Error completing email:", error);
        }
    };

    const handleProlificSubmit = async (e) => {
        e.preventDefault();
        if (prolificIdInput.trim()) {
            try {
                const response = await axios.post(`${API_URL}/auth/login`, {
                    prolific_id: prolificIdInput.trim()
                });

                const newParticipantId = response.data.participant_id;
                localStorage.setItem('participant_id', newParticipantId);
                setParticipantId(newParticipantId);
                setShowIdPrompt(false);
            } catch (error) {
                console.error("Login failed:", error);
                alert("Failed to start simulation. Please try again.");
            }
        }
    };

    const handleReset = () => {
        if (confirm("Reset simulation? This will clear your ID and reload.")) {
            localStorage.removeItem('participant_id');
            window.location.reload();
        }
    };

    const selectedEmail = emailList.find(e => e.id === selectedEmailId) || null;
    // Check if the selected email is the latest one (first in the list)
    // If it is, actions are enabled. If not (viewing history), actions are disabled.
    const isLatest = activeFolder === 'inbox' && emailList.length > 0 && selectedEmailId === emailList[0].id;

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

    if (loading && emailList.length === 0 && !finished) {
        return (
            <div className="flex h-screen items-center justify-center bg-gray-100">
                <div className="flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <div className="text-blue-600 font-semibold">Loading Outlook...</div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-white overflow-hidden font-sans">
            {/* OWA Header */}
            <div className="absolute top-0 left-0 w-full h-[48px] bg-[#0078d4] flex items-center justify-between px-2 z-50 text-white select-none">
                <div className="flex items-center">
                    {/* Waffle Icon */}
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

            {/* Main Content Area (below header) */}
            <div className="flex w-full h-full pt-[48px]">
                {/* Left Rail (App Switcher) */}
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

                {finished ? (
                    <>
                        <EmailList
                            emails={emailList}
                            selectedId={null}
                            onSelect={() => { }}
                            onDone={() => { }}
                            actionsTaken={false}
                        />
                        <ReadingPane
                            email={null}
                            onAction={() => { }}
                            isLatest={false}
                            actionsTaken={false}
                            onDone={() => { }}
                            isFinished={true}
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
