/**
 * CYPEARL Application - Main App Component
 * 
 * Two User Paths:
 * 1. ADMIN PATH (Research): Phase 1 → Phase 2 → Phase 3 → Phase 4
 * 2. CISO PATH (Enterprise): Upload → Match → Select → Test → Report
 * 
 * Features:
 * - Landing page with path selection (Admin vs CISO)
 * - Admin: Full research pipeline with phase progression
 * - CISO: Simplified enterprise workflow
 * - State persistence across sessions
 */

import React, { useState, useEffect } from 'react';
import {
  Users, Brain, ArrowRight, CheckCircle, AlertCircle,
  Database, Cpu, BarChart3, Sparkles, Shield, Target,
  ChevronLeft, Info, Building, FlaskConical, Lock,
  Upload, Mail, Play, FileText, Settings, Beaker
} from 'lucide-react';
import Phase1Dashboard from './components/Phase1Dashboard';
import Phase2Dashboard from './components/Phase2Dashboard';
import CISODashboard from './components/CISODashboard';
import './App.css';

// ============================================================================
// LANDING PAGE COMPONENT
// ============================================================================

const LandingPage = ({
  onSelectPath,
  onSelectPhase,
  phase1Complete,
  phase2Complete,
  exportedPersonas
}) => {
  const [selectedPath, setSelectedPath] = useState(null); // 'admin' or 'ciso'

  // Admin Research Phases
  const adminPhases = [
    {
      id: 1,
      title: 'Phase 1: Persona Discovery',
      description: 'Discover and validate behavioral personas from phishing study data through clustering analysis and expert validation.',
      icon: Users,
      color: 'indigo',
      features: [
        'Dataset exploration & quality analysis',
        'Multi-algorithm clustering optimization',
        'Persona characterization & naming',
        'Statistical validation (η² analysis)',
        'Expert Delphi validation',
        'AI-ready persona export'
      ],
      status: phase1Complete ? 'complete' : 'available',
      cta: phase1Complete ? 'Review Phase 1' : 'Start Phase 1'
    },
    {
      id: 2,
      title: 'Phase 2: LLM Calibration',
      description: 'Test all LLM × Prompt × Email combinations to find optimal configurations for each persona.',
      icon: Brain,
      color: 'purple',
      features: [
        'Multi-provider LLM testing',
        'Prompt configuration optimization',
        'Fidelity analysis & scoring',
        'Cost-performance curves',
        'Configuration validation',
        'Publish best configs'
      ],
      status: phase2Complete ? 'complete' : (exportedPersonas?.personas?.length > 0 ? 'ready' : 'available'),
      cta: phase2Complete ? 'Review Phase 2' : 'Start Phase 2',
      note: !phase1Complete && !exportedPersonas?.personas?.length
        ? 'Complete Phase 1 first to transfer personas.'
        : null
    },
    {
      id: 3,
      title: 'Phase 3: Validation',
      description: 'Validate AI persona fidelity against new human participant data.',
      icon: BarChart3,
      color: 'green',
      features: [
        'New data collection',
        'Independent sample testing',
        'Fidelity confirmation',
        'Boundary condition mapping'
      ],
      status: 'coming_soon',
      cta: 'Coming Soon',
      disabled: true
    },
    {
      id: 4,
      title: 'Phase 4: Deployment',
      description: 'Deploy validated personas and configurations to production.',
      icon: Cpu,
      color: 'blue',
      features: [
        'Configuration publishing',
        'API deployment',
        'Continuous monitoring',
        'Version management'
      ],
      status: 'coming_soon',
      cta: 'Coming Soon',
      disabled: true
    }
  ];

  // If no path selected, show path selection
  if (!selectedPath) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                  <Shield className="text-white" size={24} />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">CYPEARL</h1>
                  <p className="text-xs text-gray-500">Cyber Personas for AI Research Lab</p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="max-w-6xl mx-auto px-6 py-16">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Welcome to CYPEARL
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              AI-powered behavioral persona platform for phishing susceptibility research and enterprise security testing.
            </p>
          </div>

          {/* Path Selection */}
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Admin Path */}
            <div
              onClick={() => setSelectedPath('admin')}
              className="relative bg-white rounded-2xl border-2 border-gray-200 p-8 cursor-pointer transition-all duration-300 hover:shadow-xl hover:scale-[1.02] hover:border-indigo-300"
            >
              <div className="absolute -top-3 left-6">
                <span className="bg-indigo-600 text-white px-3 py-1 rounded-full text-xs font-medium">
                  Research
                </span>
              </div>

              <div className="flex items-start gap-4 mb-6">
                <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center justify-center">
                  <FlaskConical className="text-indigo-600" size={32} />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">Admin Dashboard</h3>
                  <p className="text-gray-600">
                    Full research pipeline for discovering personas, calibrating LLMs,
                    and validating AI fidelity.
                  </p>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center">
                    <span className="text-indigo-600 font-bold">1</span>
                  </div>
                  <span>Discover personas from participant data</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center">
                    <span className="text-indigo-600 font-bold">2</span>
                  </div>
                  <span>Test ALL LLM × Prompt combinations</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center">
                    <span className="text-indigo-600 font-bold">3</span>
                  </div>
                  <span>Validate and publish best configurations</span>
                </div>
              </div>

              <div className="p-3 bg-indigo-50 rounded-lg text-sm text-indigo-700 mb-6">
                <strong>For researchers:</strong> Run ~150,000 API calls to find optimal persona configurations
              </div>

              <button className="w-full py-3 bg-indigo-600 text-white rounded-xl font-medium hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2">
                Enter Admin Dashboard
                <ArrowRight size={18} />
              </button>
            </div>

            {/* CISO Path */}
            <div
              onClick={() => onSelectPath('ciso')}
              className="relative bg-white rounded-2xl border-2 border-gray-200 p-8 cursor-pointer transition-all duration-300 hover:shadow-xl hover:scale-[1.02] hover:border-green-300"
            >
              <div className="absolute -top-3 left-6">
                <span className="bg-green-600 text-white px-3 py-1 rounded-full text-xs font-medium">
                  Enterprise
                </span>
              </div>

              <div className="flex items-start gap-4 mb-6">
                <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center">
                  <Building className="text-green-600" size={32} />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">CISO Dashboard</h3>
                  <p className="text-gray-600">
                    Simplified workflow for enterprise security teams to test phishing
                    susceptibility using validated AI personas.
                  </p>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-green-50 rounded-lg flex items-center justify-center">
                    <Upload className="text-green-600" size={16} />
                  </div>
                  <span>Upload employee assessment data</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-green-50 rounded-lg flex items-center justify-center">
                    <Users className="text-green-600" size={16} />
                  </div>
                  <span>Auto-match to validated personas</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-green-50 rounded-lg flex items-center justify-center">
                    <Mail className="text-green-600" size={16} />
                  </div>
                  <span>Test custom phishing emails</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-600">
                  <div className="w-8 h-8 bg-green-50 rounded-lg flex items-center justify-center">
                    <BarChart3 className="text-green-600" size={16} />
                  </div>
                  <span>Get behavioral predictions & reports</span>
                </div>
              </div>

              <div className="p-3 bg-green-50 rounded-lg text-sm text-green-700 mb-6">
                <strong>For CISOs:</strong> Use pre-validated configs with ~200 API calls per test
              </div>

              <button className="w-full py-3 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 transition-colors flex items-center justify-center gap-2">
                Enter CISO Dashboard
                <ArrowRight size={18} />
              </button>
            </div>
          </div>

          {/* How It Works */}
          <div className="mt-16 bg-white rounded-2xl border border-gray-200 p-8">
            <h3 className="text-lg font-bold text-gray-900 mb-6 text-center">
              How CYPEARL Works
            </h3>
            <div className="grid md:grid-cols-2 gap-8">
              {/* Admin Flow */}
              <div>
                <h4 className="font-semibold text-indigo-600 mb-4 flex items-center gap-2">
                  <FlaskConical size={18} />
                  Admin (Research) Flow
                </h4>
                <div className="space-y-2 text-sm text-gray-600">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600">1</div>
                    <span>Collect human participant phishing data</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600">2</div>
                    <span>Cluster into behavioral personas</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600">3</div>
                    <span>Test ALL LLM × Prompt combinations</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600">4</div>
                    <span>Find best config for each persona</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600">5</div>
                    <span>Publish validated configurations</span>
                  </div>
                </div>
              </div>

              {/* CISO Flow */}
              <div>
                <h4 className="font-semibold text-green-600 mb-4 flex items-center gap-2">
                  <Building size={18} />
                  CISO (Enterprise) Flow
                </h4>
                <div className="space-y-2 text-sm text-gray-600">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">1</div>
                    <span>Upload employee assessment data</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">2</div>
                    <span>Auto-match to existing personas</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">3</div>
                    <span>Accept recommended LLM configs</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">4</div>
                    <span>Create/upload test emails</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">5</div>
                    <span>Get predictions & recommendations</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between text-sm text-gray-500">
              <span>CYPEARL Research Platform</span>
              <span>Phishing Susceptibility & AI Fidelity Research</span>
            </div>
          </div>
        </footer>
      </div>
    );
  }

  // Admin path selected - show phase selection
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSelectedPath(null)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ChevronLeft size={20} />
              </button>
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Shield className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">CYPEARL Admin</h1>
                <p className="text-xs text-gray-500">Research Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <FlaskConical className="text-indigo-600" size={18} />
              <span className="text-sm text-indigo-600 font-medium">Research Mode</span>
            </div>
          </div>
        </div>
      </header>

      {/* Phase Selection */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Research Pipeline
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Complete each phase to build, calibrate, and validate AI personas
            that accurately replicate human phishing susceptibility behavior.
          </p>
        </div>

        {/* Phase Cards Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {adminPhases.map((phase) => {
            const Icon = phase.icon;
            const isComplete = phase.status === 'complete';
            const isReady = phase.status === 'ready';
            const isDisabled = phase.disabled;

            return (
              <div
                key={phase.id}
                className={`
                                    relative bg-white rounded-2xl border-2 transition-all duration-300
                                    ${isDisabled
                    ? 'opacity-60 cursor-not-allowed border-gray-200'
                    : 'hover:shadow-xl hover:scale-[1.01] cursor-pointer'
                  }
                                    ${isComplete
                    ? 'border-green-300'
                    : isReady
                      ? 'border-indigo-300'
                      : 'border-gray-200 hover:border-gray-300'
                  }
                                `}
                onClick={() => !isDisabled && onSelectPhase(phase.id)}
              >
                {/* Status Badge */}
                {isComplete && (
                  <div className="absolute -top-3 -right-3 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 shadow-lg">
                    <CheckCircle size={14} />
                    Complete
                  </div>
                )}
                {isReady && !isComplete && (
                  <div className="absolute -top-3 -right-3 bg-indigo-500 text-white px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 shadow-lg">
                    <Sparkles size={14} />
                    Ready
                  </div>
                )}
                {phase.status === 'coming_soon' && (
                  <div className="absolute -top-3 -right-3 bg-gray-400 text-white px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 shadow-lg">
                    <Lock size={14} />
                    Coming Soon
                  </div>
                )}

                <div className="p-6">
                  {/* Header */}
                  <div className="flex items-start gap-4 mb-4">
                    <div className={`
                                            w-12 h-12 rounded-xl flex items-center justify-center
                                            ${isComplete
                        ? 'bg-green-100 text-green-600'
                        : isDisabled
                          ? 'bg-gray-100 text-gray-400'
                          : `bg-${phase.color}-100 text-${phase.color}-600`
                      }
                                        `}>
                      <Icon size={24} />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-gray-900 mb-1">
                        {phase.title}
                      </h3>
                      <p className="text-sm text-gray-600">
                        {phase.description}
                      </p>
                    </div>
                  </div>

                  {/* Features */}
                  <div className="grid grid-cols-2 gap-2 mb-4">
                    {phase.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs text-gray-600">
                        <CheckCircle
                          size={12}
                          className={isComplete ? 'text-green-500' : 'text-gray-300'}
                        />
                        {feature}
                      </div>
                    ))}
                  </div>

                  {/* Note */}
                  {phase.note && (
                    <div className="mb-4 p-2 bg-amber-50 border border-amber-200 rounded-lg">
                      <div className="flex items-start gap-2">
                        <Info size={14} className="text-amber-600 mt-0.5 flex-shrink-0" />
                        <p className="text-xs text-amber-700">{phase.note}</p>
                      </div>
                    </div>
                  )}

                  {/* CTA Button */}
                  <button
                    disabled={isDisabled}
                    className={`
                                            w-full py-2.5 px-4 rounded-xl font-medium text-sm
                                            flex items-center justify-center gap-2 transition-colors
                                            ${isDisabled
                        ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                        : isComplete
                          ? 'bg-green-600 text-white hover:bg-green-700'
                          : 'bg-indigo-600 text-white hover:bg-indigo-700'
                      }
                                        `}
                  >
                    {phase.cta}
                    {!isDisabled && <ArrowRight size={16} />}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Pipeline Visualization */}
        <div className="bg-white rounded-2xl border border-gray-200 p-6">
          <h3 className="text-sm font-bold text-gray-900 mb-4 text-center">
            Research Pipeline Flow
          </h3>
          <div className="flex items-center justify-center gap-3 overflow-x-auto pb-2">
            {[
              { icon: Database, label: 'Data', phase: 1 },
              { icon: Target, label: 'Cluster', phase: 1 },
              { icon: Users, label: 'Personas', phase: 1 },
              { icon: Sparkles, label: 'Export', phase: 1 },
              { icon: Cpu, label: 'LLMs', phase: 2 },
              { icon: Settings, label: 'Calibrate', phase: 2 },
              { icon: BarChart3, label: 'Validate', phase: 3 },
              { icon: CheckCircle, label: 'Deploy', phase: 4 },
            ].map((step, idx, arr) => (
              <React.Fragment key={step.label}>
                <div className="flex flex-col items-center gap-1 min-w-[60px]">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${step.phase === 1 && phase1Complete ? 'bg-green-100 text-green-600' :
                    step.phase === 2 && phase2Complete ? 'bg-green-100 text-green-600' :
                      'bg-gray-100 text-gray-400'
                    }`}>
                    <step.icon size={20} />
                  </div>
                  <span className="text-xs text-gray-500">{step.label}</span>
                </div>
                {idx < arr.length - 1 && (
                  <ArrowRight className="text-gray-300 flex-shrink-0" size={16} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>CYPEARL Admin Dashboard</span>
            <button
              onClick={() => setSelectedPath(null)}
              className="text-indigo-600 hover:text-indigo-700"
            >
              Switch to CISO Dashboard
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================

const App = () => {
  // Current view: 'landing', 'phase1', 'phase2', 'ciso'
  const [currentView, setCurrentView] = useState('landing');

  // Phase completion states
  const [phase1Complete, setPhase1Complete] = useState(false);
  const [phase2Complete, setPhase2Complete] = useState(false);

  // Exported data from Phase 1 for Phase 2
  const [exportedPersonas, setExportedPersonas] = useState(null);

  // Load saved state from localStorage on mount
  useEffect(() => {
    try {
      const savedState = localStorage.getItem('cypearl_state');
      if (savedState) {
        const state = JSON.parse(savedState);
        if (state.phase1Complete) setPhase1Complete(state.phase1Complete);
        if (state.phase2Complete) setPhase2Complete(state.phase2Complete);
        if (state.exportedPersonas) setExportedPersonas(state.exportedPersonas);
      }
    } catch (e) {
      console.warn('Failed to load saved state:', e);
    }
  }, []);

  // Save state to localStorage when it changes
  useEffect(() => {
    try {
      const stateToSave = {
        phase1Complete,
        phase2Complete,
        exportedPersonas
      };
      localStorage.setItem('cypearl_state', JSON.stringify(stateToSave));
    } catch (e) {
      console.warn('Failed to save state:', e);
    }
  }, [phase1Complete, phase2Complete, exportedPersonas]);

  // Handle path selection (admin vs ciso)
  const handleSelectPath = (path) => {
    if (path === 'ciso') {
      setCurrentView('ciso');
    }
  };

  // Handle phase selection from landing page
  const handleSelectPhase = (phaseId) => {
    if (phaseId === 1) {
      setCurrentView('phase1');
    } else if (phaseId === 2) {
      setCurrentView('phase2');
    }
  };

  // Handle export from Phase 1 to Phase 2
  const handleExportToPhase2 = (exportData) => {
    console.log('Received export data from Phase 1:', exportData);
    setExportedPersonas(exportData);
  };

  // Handle Phase 1 completion
  const handlePhase1Complete = () => {
    setPhase1Complete(true);
    setCurrentView('phase2');
  };

  // Handle Phase 2 completion
  const handlePhase2Complete = () => {
    setPhase2Complete(true);
    setCurrentView('landing');
  };

  // Handle back navigation
  const handleBackToLanding = () => {
    setCurrentView('landing');
  };

  // Render CISO Dashboard
  if (currentView === 'ciso') {
    return <CISODashboard onBack={handleBackToLanding} />;
  }

  // Render Phase 1
  if (currentView === 'phase1') {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="bg-white border-b px-4 py-2">
          <button
            onClick={handleBackToLanding}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ChevronLeft size={18} />
            Back to Overview
          </button>
        </div>
        <Phase1Dashboard
          onExportToPhase2={handleExportToPhase2}
          onPhaseComplete={handlePhase1Complete}
        />
      </div>
    );
  }

  // Render Phase 2
  if (currentView === 'phase2') {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="bg-white border-b px-4 py-2">
          <button
            onClick={handleBackToLanding}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ChevronLeft size={18} />
            Back to Overview
          </button>
        </div>
        <Phase2Dashboard
          importedPersonas={exportedPersonas}
          onPhaseComplete={handlePhase2Complete}
        />
      </div>
    );
  }

  // Default: Landing page
  return (
    <LandingPage
      onSelectPath={handleSelectPath}
      onSelectPhase={handleSelectPhase}
      phase1Complete={phase1Complete}
      phase2Complete={phase2Complete}
      exportedPersonas={exportedPersonas}
    />
  );
};

export default App;