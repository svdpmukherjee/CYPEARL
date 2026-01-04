import React, { useState, useEffect } from 'react';
import { Flag, Trash2, Reply, ReplyAll, Forward } from 'lucide-react';

const EmailView = ({ email, onAction }) => {
    const [hoverStart, setHoverStart] = useState(null);

    const handleMouseEnter = (type) => {
        setHoverStart(Date.now());
    };

    const handleMouseLeave = (type) => {
        if (hoverStart) {
            const duration = Date.now() - hoverStart;
            console.log(`Hovered on ${type} for ${duration}ms`);
            // In a real app, we'd accumulate this data to send with the action
        }
        setHoverStart(null);
    };

    if (!email) {
        return (
            <div className="flex-1 flex items-center justify-center text-gray-500 bg-white">
                Select an email to read
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col bg-white h-screen overflow-hidden">
            {/* Toolbar */}
            <div className="h-12 border-b border-gray-200 flex items-center px-4 space-x-4 bg-gray-50">
                <button
                    onClick={() => onAction('delete')}
                    className="p-1.5 hover:bg-gray-200 rounded text-gray-600 flex items-center space-x-1"
                >
                    <Trash2 size={16} />
                    <span className="text-sm">Delete</span>
                </button>
                <div className="h-6 w-px bg-gray-300 mx-2"></div>
                <button
                    onClick={() => onAction('report')}
                    className="p-1.5 hover:bg-gray-200 rounded text-red-600 flex items-center space-x-1"
                >
                    <Flag size={16} />
                    <span className="text-sm">Report Phishing</span>
                </button>
            </div>

            {/* Email Header */}
            <div className="p-6 border-b border-gray-100">
                <h1 className="text-2xl font-semibold mb-4">{email.subject}</h1>
                <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-lg">
                        {email.sender_name.charAt(0)}
                    </div>
                    <div className="flex-1">
                        <div
                            className="font-semibold text-gray-900 cursor-pointer hover:underline decoration-dotted"
                            onMouseEnter={() => handleMouseEnter('sender')}
                            onMouseLeave={() => handleMouseLeave('sender')}
                        >
                            {email.sender_name} <span className="text-gray-500 font-normal">&lt;{email.sender_email}&gt;</span>
                        </div>
                        <div className="text-sm text-gray-500 mt-0.5">
                            To: You
                        </div>
                    </div>
                    <div className="text-sm text-gray-500">
                        {new Date(email.timestamp).toLocaleString()}
                    </div>
                </div>
            </div>

            {/* Email Body */}
            <div className="flex-1 p-8 overflow-y-auto">
                <div
                    className="prose max-w-none"
                    dangerouslySetInnerHTML={{ __html: email.body }}
                    onMouseEnter={(e) => {
                        if (e.target.tagName === 'A') handleMouseEnter('link');
                    }}
                    onMouseLeave={(e) => {
                        if (e.target.tagName === 'A') handleMouseLeave('link');
                    }}
                    onClick={(e) => {
                        if (e.target.tagName === 'A') {
                            e.preventDefault();
                            onAction('click');
                        }
                    }}
                />
            </div>
        </div>
    );
};

export default EmailView;
