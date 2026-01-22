/**
 * Phase 2 Execution Tab
 *
 * Run experiments with real-time progress monitoring.
 * Includes detailed logging similar to CalibrationTab.
 */

import React, { useState, useEffect } from 'react';
import {
    Play, Pause, StopCircle, RefreshCw, CheckCircle, XCircle,
    Clock, DollarSign, Zap, AlertTriangle, List, Download,
    FileText, Eye, ChevronDown, ChevronRight, Copy, Check
} from 'lucide-react';
import * as api from '../../services/phase2Api';

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const ExecutionTab = ({
    experiments,
    currentExperiment,
    progress,
    onSelectExperiment,
    onRunExperiment,
    onStopExperiment
}) => {
    const [showTrials, setShowTrials] = useState(false);
    const [resumeFromCheckpoint, setResumeFromCheckpoint] = useState(false);
    const [showModelBreakdown, setShowModelBreakdown] = useState(true);
    const [logContent, setLogContent] = useState(null);
    const [loadingLog, setLoadingLog] = useState(false);
    const [showLogModal, setShowLogModal] = useState(false);
    const [copied, setCopied] = useState(false);

    // Load log for completed experiment
    const loadLog = async (experimentId) => {
        setLoadingLog(true);
        try {
            const result = await api.getExperimentLogs(experimentId);
            setLogContent(result);
            setShowLogModal(true);
        } catch (error) {
            console.error('Failed to load logs:', error);
            // Try to generate log if not found
            try {
                await api.generateExperimentLog(experimentId);
                const result = await api.getExperimentLogs(experimentId);
                setLogContent(result);
                setShowLogModal(true);
            } catch (genError) {
                alert('Failed to load or generate logs: ' + genError.message);
            }
        }
        setLoadingLog(false);
    };

    // Download log file
    const downloadLog = () => {
        if (!logContent) return;
        const blob = new Blob([logContent.log_content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = logContent.file_name || 'experiment_log.txt';
        a.click();
        URL.revokeObjectURL(url);
    };

    // Copy log to clipboard
    const copyLog = () => {
        if (!logContent) return;
        navigator.clipboard.writeText(logContent.log_content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // Calculate model breakdown from recent trials
    const modelBreakdown = React.useMemo(() => {
        if (!progress?.recent_trials) return [];
        const byModel = {};
        progress.recent_trials.forEach(trial => {
            if (!byModel[trial.model_id]) {
                byModel[trial.model_id] = { total: 0, success: 0, clicked: 0 };
            }
            byModel[trial.model_id].total++;
            if (trial.success) byModel[trial.model_id].success++;
            if (trial.action === 'click') byModel[trial.model_id].clicked++;
        });
        return Object.entries(byModel).map(([model, data]) => ({
            model,
            ...data,
            successRate: data.total > 0 ? (data.success / data.total * 100).toFixed(0) : 0,
            clickRate: data.total > 0 ? (data.clicked / data.total * 100).toFixed(0) : 0
        }));
    }, [progress?.recent_trials]);

    // Get experiments ready to run
    const readyExperiments = experiments.filter(e =>
        e.status === 'draft' || e.status === 'ready' || e.status === 'paused'
    );
    const runningExperiment = experiments.find(e => e.status === 'running');
    const completedExperiments = experiments.filter(e => e.status === 'completed');

    // Calculate progress percentage
    const progressPercent = progress
        ? ((progress.completed_trials / progress.total_trials) * 100).toFixed(1)
        : 0;

    // Handle run
    const handleRun = async (experimentId) => {
        await onRunExperiment(experimentId, { resume_from_checkpoint: resumeFromCheckpoint });
    };

    return (
        <div className="space-y-6">
            {/* Running Experiment */}
            {(runningExperiment || progress) && (
                <div className="bg-white rounded-xl border border-purple-200 p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <Zap className="text-purple-600 animate-pulse" size={20} />
                            Running: {runningExperiment?.name || currentExperiment?.name}
                        </h3>
                        <button
                            onClick={() => onStopExperiment(runningExperiment?.experiment_id || currentExperiment?.experiment_id)}
                            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm hover:bg-red-200 flex items-center gap-2"
                        >
                            <StopCircle size={16} />
                            Stop
                        </button>
                    </div>

                    {/* Progress Bar */}
                    <div className="mb-4">
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-600">
                                {progress?.completed_trials?.toLocaleString() || 0} / {progress?.total_trials?.toLocaleString() || 0} trials
                            </span>
                            <span className="font-medium text-purple-600">{progressPercent}%</span>
                        </div>
                        <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-purple-500 to-purple-600 transition-all duration-300"
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <StatBox
                            icon={<CheckCircle className="text-green-500" size={16} />}
                            label="Completed"
                            value={progress?.completed_trials?.toLocaleString() || 0}
                        />
                        <StatBox
                            icon={<XCircle className="text-red-500" size={16} />}
                            label="Failed"
                            value={progress?.failed_trials || 0}
                        />
                        <StatBox
                            icon={<DollarSign className="text-green-600" size={16} />}
                            label="Cost"
                            value={`$${(progress?.total_cost || 0).toFixed(2)}`}
                        />
                        <StatBox
                            icon={<Clock className="text-blue-500" size={16} />}
                            label="Elapsed"
                            value={formatDuration(progress?.elapsed_seconds || 0)}
                        />
                        <StatBox
                            icon={<Clock className="text-amber-500" size={16} />}
                            label="Remaining"
                            value={formatDuration(progress?.estimated_remaining || 0)}
                        />
                    </div>

                    {/* Real-time Model Breakdown */}
                    {modelBreakdown.length > 0 && (
                        <div className="mt-4 pt-4 border-t">
                            <button
                                onClick={() => setShowModelBreakdown(!showModelBreakdown)}
                                className="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1 mb-3"
                            >
                                {showModelBreakdown ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                Real-time Model Performance (last {progress?.recent_trials?.length || 0} trials)
                            </button>
                            {showModelBreakdown && (
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                    {modelBreakdown.map((m, i) => (
                                        <div key={i} className="p-3 bg-gray-50 rounded-lg border">
                                            <div className="font-medium text-sm truncate" title={m.model}>
                                                {m.model}
                                            </div>
                                            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                                                <div>
                                                    <span className="text-gray-500">Trials:</span>
                                                    <span className="ml-1 font-medium">{m.total}</span>
                                                </div>
                                                <div>
                                                    <span className="text-gray-500">Parse:</span>
                                                    <span className={`ml-1 font-medium ${m.successRate >= 90 ? 'text-green-600' : 'text-amber-600'}`}>
                                                        {m.successRate}%
                                                    </span>
                                                </div>
                                                <div className="col-span-2">
                                                    <span className="text-gray-500">Click Rate:</span>
                                                    <span className="ml-1 font-medium text-purple-600">{m.clickRate}%</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Recent Trials */}
                    {progress?.recent_trials && progress.recent_trials.length > 0 && (
                        <div className="mt-4 pt-4 border-t">
                            <button
                                onClick={() => setShowTrials(!showTrials)}
                                className="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1 mb-2"
                            >
                                {showTrials ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                {showTrials ? 'Hide' : 'Show'} Recent Trials Log
                            </button>
                            {showTrials && (
                                <div className="max-h-48 overflow-y-auto space-y-1 font-mono text-xs">
                                    {progress.recent_trials.map((trial, i) => (
                                        <div
                                            key={i}
                                            className={`p-2 rounded flex items-center justify-between ${trial.success ? 'bg-green-50' : 'bg-red-50'
                                                }`}
                                        >
                                            <span className="flex items-center gap-2">
                                                <span className="text-gray-400">{String(i + 1).padStart(3, '0')}</span>
                                                <span className="text-blue-600">{trial.model_id}</span>
                                                <span className="text-gray-400">|</span>
                                                <span>{trial.persona_id}</span>
                                                <span className="text-gray-400">|</span>
                                                <span className="text-purple-600">{trial.email_id}</span>
                                            </span>
                                            <span className={`px-2 py-0.5 rounded ${
                                                trial.action === 'click' ? 'bg-red-100 text-red-700' :
                                                trial.action === 'report' ? 'bg-green-100 text-green-700' :
                                                trial.action === 'ignore' ? 'bg-gray-100 text-gray-700' :
                                                'bg-amber-100 text-amber-700'
                                            }`}>
                                                {trial.action || 'error'}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Experiment Queue */}
            <div className="bg-white rounded-xl border p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <List className="text-blue-600" size={20} />
                    Experiment Queue
                </h3>

                {readyExperiments.length > 0 ? (
                    <div className="space-y-3">
                        {readyExperiments.map(exp => (
                            <div
                                key={exp.experiment_id}
                                className={`p-4 rounded-lg border ${currentExperiment?.experiment_id === exp.experiment_id
                                        ? 'border-purple-300 bg-purple-50'
                                        : 'border-gray-200 hover:bg-gray-50'
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div>
                                        <div className="font-medium">{exp.name}</div>
                                        <div className="text-sm text-gray-500">
                                            {exp.total_trials?.toLocaleString() || 0} trials •
                                            {exp.persona_ids?.length || 0} personas •
                                            {exp.model_ids?.length || 0} models
                                        </div>
                                        {exp.description && (
                                            <div className="text-xs text-gray-400 mt-1">{exp.description}</div>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <StatusBadge status={exp.status} />
                                        {exp.status !== 'running' && (
                                            <button
                                                onClick={() => {
                                                    onSelectExperiment(exp);
                                                    handleRun(exp.experiment_id);
                                                }}
                                                disabled={runningExperiment}
                                                className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
                                            >
                                                <Play size={16} />
                                                Run
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* Expand details */}
                                {currentExperiment?.experiment_id === exp.experiment_id && (
                                    <div className="mt-4 pt-4 border-t border-gray-200">
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                            <div>
                                                <div className="text-gray-500">Personas</div>
                                                <div className="font-medium">{exp.persona_ids?.join(', ')}</div>
                                            </div>
                                            <div>
                                                <div className="text-gray-500">Models</div>
                                                <div className="font-medium">{exp.model_ids?.length} selected</div>
                                            </div>
                                            <div>
                                                <div className="text-gray-500">Prompts</div>
                                                <div className="font-medium">{exp.prompt_configs?.join(', ')}</div>
                                            </div>
                                            <div>
                                                <div className="text-gray-500">Estimated Cost</div>
                                                <div className="font-medium text-green-600">
                                                    ${exp.estimated_cost?.toFixed(2) || 'N/A'}
                                                </div>
                                            </div>
                                        </div>

                                        {exp.status === 'paused' && (
                                            <div className="mt-4 flex items-center gap-3">
                                                <label className="flex items-center gap-2 text-sm">
                                                    <input
                                                        type="checkbox"
                                                        checked={resumeFromCheckpoint}
                                                        onChange={(e) => setResumeFromCheckpoint(e.target.checked)}
                                                        className="w-4 h-4 accent-purple-600"
                                                    />
                                                    Resume from checkpoint
                                                </label>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-8 text-gray-500">
                        <List className="mx-auto mb-2 text-gray-300" size={32} />
                        <p>No experiments ready to run</p>
                        <p className="text-sm">Create an experiment in the Builder tab</p>
                    </div>
                )}
            </div>

            {/* Completed Experiments */}
            {completedExperiments.length > 0 && (
                <div className="bg-white rounded-xl border p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <CheckCircle className="text-green-600" size={20} />
                        Completed Experiments ({completedExperiments.length})
                    </h3>
                    <div className="space-y-2">
                        {completedExperiments.map(exp => (
                            <div
                                key={exp.experiment_id}
                                className="p-3 rounded-lg border border-green-200 bg-green-50 flex items-center justify-between"
                            >
                                <div>
                                    <div className="font-medium">{exp.name}</div>
                                    <div className="text-sm text-gray-600">
                                        {exp.completed_trials?.toLocaleString()} trials •
                                        ${exp.actual_cost?.toFixed(2) || '0.00'}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-500">
                                        {formatDate(exp.completed_at)}
                                    </span>
                                    <button
                                        onClick={() => loadLog(exp.experiment_id)}
                                        disabled={loadingLog}
                                        className="px-3 py-1 bg-white border rounded text-sm hover:bg-gray-50 flex items-center gap-1"
                                    >
                                        <FileText size={14} />
                                        {loadingLog ? 'Loading...' : 'View Logs'}
                                    </button>
                                    <button
                                        onClick={() => onSelectExperiment(exp)}
                                        className="px-3 py-1 bg-white border rounded text-sm hover:bg-gray-50"
                                    >
                                        View Results
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Log Modal */}
            {showLogModal && logContent && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-xl max-w-5xl w-full max-h-[90vh] flex flex-col">
                        <div className="p-4 border-b flex items-center justify-between">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <FileText className="text-purple-600" size={20} />
                                Experiment Log: {logContent.file_name}
                            </h3>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={copyLog}
                                    className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center gap-1"
                                >
                                    {copied ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
                                    {copied ? 'Copied!' : 'Copy'}
                                </button>
                                <button
                                    onClick={downloadLog}
                                    className="px-3 py-1.5 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg text-sm flex items-center gap-1"
                                >
                                    <Download size={14} />
                                    Download
                                </button>
                                <button
                                    onClick={() => setShowLogModal(false)}
                                    className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 overflow-auto p-4">
                            <pre className="text-xs font-mono whitespace-pre-wrap bg-gray-50 p-4 rounded-lg border max-h-[70vh] overflow-auto">
                                {logContent.log_content}
                            </pre>
                        </div>
                    </div>
                </div>
            )}

            {/* Execution Tips */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-start gap-3">
                    <AlertTriangle className="text-blue-600 mt-0.5" size={20} />
                    <div>
                        <h4 className="font-semibold text-blue-900">Execution Tips</h4>
                        <ul className="text-sm text-blue-700 mt-1 space-y-1">
                            <li>• Experiments run in the background - you can navigate to other tabs</li>
                            <li>• Progress is automatically checkpointed every 50 trials</li>
                            <li>• Rate limits are respected per provider to avoid API errors</li>
                            <li>• Failed trials are logged but don't stop the experiment</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

const StatBox = ({ icon, label, value }) => (
    <div className="p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            {icon}
            {label}
        </div>
        <div className="text-xl font-bold">{value}</div>
    </div>
);

const StatusBadge = ({ status }) => {
    const styles = {
        draft: 'bg-gray-100 text-gray-600',
        ready: 'bg-blue-100 text-blue-700',
        running: 'bg-purple-100 text-purple-700',
        paused: 'bg-yellow-100 text-yellow-700',
        completed: 'bg-green-100 text-green-700',
        failed: 'bg-red-100 text-red-700',
    };

    return (
        <span className={`px-2 py-1 rounded text-xs font-medium ${styles[status] || styles.draft}`}>
            {status}
        </span>
    );
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDuration(seconds) {
    if (!seconds || seconds <= 0) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins >= 60) {
        const hrs = Math.floor(mins / 60);
        const remainMins = mins % 60;
        return `${hrs}h ${remainMins}m`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default ExecutionTab;