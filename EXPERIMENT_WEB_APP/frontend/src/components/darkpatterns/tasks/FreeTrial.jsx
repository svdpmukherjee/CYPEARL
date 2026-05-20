/**
 * FreeTrial Task Component
 *
 * Dark Pattern Version (dp_03):
 * - Requires credit card
 * - Auto-renewal in tiny text
 * - Annual plan preselected
 *
 * Clean UI Version (clean_03):
 * - No card required
 * - Clear terms
 * - No preselection
 */

import React, { useState } from 'react';
import { CreditCard, Check, Star, Sparkles, AlertCircle } from 'lucide-react';

export default function FreeTrial({ task, uiContent, onAction, trackClick, trackHover, trackExpand }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const serviceName = content.service_name || 'StreamMax Pro';
  const headline = content.headline || (isDark ? 'Start Your FREE Trial' : 'Try Premium Free');
  const startTrialButton = content.start_trial_button || 'Start Free Trial';
  const skipButton = content.skip_button || 'Maybe Later';

  const [selectedPlan, setSelectedPlan] = useState(isDark ? 'annual' : null);
  const [cardEntered, setCardEntered] = useState(false);

  const handleStartTrial = () => {
    trackClick('start_trial_button', 'button', !isDark);
    onAction(isDark ? 'start_trial_with_card' : 'start_trial');
  };

  const handleSkip = () => {
    trackClick('skip_trial_button', 'button', true);
    onAction('skip_trial');
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gradient-to-br from-purple-900 to-indigo-900 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-indigo-600 px-6 py-8 text-center text-white">
            <Sparkles className="w-12 h-12 mx-auto mb-3" />
            <h1 className="text-2xl font-bold mb-2">{headline}</h1>
            <p className="text-purple-100">Unlock {serviceName} Today</p>
          </div>

          {/* Plan Selection */}
          <div className="p-6">
            <p className="text-gray-600 text-sm mb-4">Select your plan:</p>

            {/* Annual Plan - Preselected & Highlighted */}
            <div
              onClick={() => {
                trackClick('annual_plan', 'radio', false);
                setSelectedPlan('annual');
              }}
              onMouseEnter={() => trackHover('annual_plan', 'option', false)}
              className={`p-4 rounded-xl mb-3 cursor-pointer transition-all ${
                selectedPlan === 'annual'
                  ? 'bg-purple-50 border-2 border-purple-500 shadow-md'
                  : 'border border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-gray-800">Annual Plan</span>
                    <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full">
                      BEST VALUE
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-gray-800 mt-1">$99.99<span className="text-sm font-normal text-gray-500">/year</span></p>
                </div>
                <div className={`w-6 h-6 rounded-full border-2 ${
                  selectedPlan === 'annual' ? 'border-purple-500 bg-purple-500' : 'border-gray-300'
                }`}>
                  {selectedPlan === 'annual' && <Check className="w-5 h-5 text-white" />}
                </div>
              </div>
            </div>

            {/* Monthly Plan - Less prominent */}
            <div
              onClick={() => {
                trackClick('monthly_plan', 'radio', true);
                setSelectedPlan('monthly');
              }}
              onMouseEnter={() => trackHover('monthly_plan', 'option', true)}
              className={`p-4 rounded-xl mb-4 cursor-pointer transition-all ${
                selectedPlan === 'monthly'
                  ? 'bg-gray-50 border-2 border-gray-400'
                  : 'border border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-medium text-gray-600">Monthly Plan</span>
                  <p className="text-lg text-gray-600 mt-1">$12.99<span className="text-sm text-gray-400">/month</span></p>
                </div>
                <div className={`w-6 h-6 rounded-full border-2 ${
                  selectedPlan === 'monthly' ? 'border-gray-400 bg-gray-400' : 'border-gray-300'
                }`}>
                  {selectedPlan === 'monthly' && <Check className="w-5 h-5 text-white" />}
                </div>
              </div>
            </div>

            {/* Credit Card Required */}
            <div className="bg-gray-50 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-2 mb-3">
                <CreditCard className="w-5 h-5 text-gray-400" />
                <span className="font-medium text-gray-700">Payment Method</span>
              </div>
              <input
                type="text"
                placeholder="Card number"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg mb-2 text-sm"
                onChange={() => {
                  trackClick('card_input', 'input', false);
                  setCardEntered(true);
                }}
              />
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="MM/YY"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
                />
                <input
                  type="text"
                  placeholder="CVC"
                  className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm"
                />
              </div>
            </div>

            {/* Start Trial Button */}
            <button
              onClick={handleStartTrial}
              onMouseEnter={() => trackHover('start_trial_button', 'button', false)}
              className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-4 rounded-xl font-bold text-lg hover:opacity-90 transition-opacity"
            >
              {startTrialButton}
            </button>

            {/* Fine Print - Very Small */}
            <p
              className="text-[9px] text-gray-300 mt-3 text-center"
              onMouseEnter={() => trackHover('fine_print', 'fine_print', false)}
            >
              {content.hidden_terms || `Trial automatically converts to paid subscription. Cancel anytime before trial ends to avoid charges.
              By clicking above, you agree to be charged ${selectedPlan === 'annual' ? '$99.99/year' : '$12.99/month'} after trial.`}
            </p>

            {/* Skip - Nearly invisible */}
            <button
              onClick={handleSkip}
              onMouseEnter={() => trackHover('skip_trial_button', 'button', true)}
              className="w-full mt-3 text-[10px] text-gray-300 hover:text-gray-400"
            >
              {skipButton}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-md w-full overflow-hidden">
        {/* Header */}
        <div className="bg-blue-50 px-6 py-6 text-center">
          <Star className="w-12 h-12 text-blue-600 mx-auto mb-3" />
          <h1 className="text-xl font-bold text-gray-800 mb-2">Try Premium Free</h1>
          <p className="text-gray-600">7-day trial, no credit card required</p>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Features */}
          <div className="space-y-3 mb-6">
            {['Unlimited access to all features', 'Ad-free experience', 'Priority support', 'Cancel anytime'].map((feature, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center">
                  <Check className="w-3 h-3 text-green-600" />
                </div>
                <span className="text-gray-700">{feature}</span>
              </div>
            ))}
          </div>

          {/* Info Box */}
          <div className="bg-blue-50 rounded-lg p-4 mb-6 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <p className="text-sm text-gray-700">
                No payment required. After your trial, choose a plan or continue with the free version.
              </p>
            </div>
          </div>

          {/* Buttons */}
          <button
            onClick={handleStartTrial}
            onMouseEnter={() => trackHover('start_trial_button', 'button', true)}
            className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors mb-3"
          >
            Start Free Trial
          </button>

          <button
            onClick={handleSkip}
            onMouseEnter={() => trackHover('skip_trial_button', 'button', false)}
            className="w-full text-gray-500 text-sm hover:text-gray-700"
          >
            No thanks, continue with free version
          </button>
        </div>
      </div>
    </div>
  );
}
