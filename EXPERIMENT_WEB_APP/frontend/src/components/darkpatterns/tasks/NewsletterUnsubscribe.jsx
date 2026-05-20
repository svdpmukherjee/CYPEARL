/**
 * NewsletterUnsubscribe Task Component
 *
 * Dark Pattern Version (dp_02):
 * - Multi-step obstruction (7 steps)
 * - Hidden final button
 * - Guilt-trip language
 *
 * Clean UI Version (clean_02):
 * - One-click unsubscribe
 */

import React, { useState, useEffect } from 'react';
import { Mail, ChevronRight, Heart, AlertTriangle, Check, X } from 'lucide-react';

export default function NewsletterUnsubscribe({ task, uiContent, onAction, trackClick, trackHover, setCurrentStep }) {
  const isDark = task.ui_type === 'dark';
  const [step, setStep] = useState(1);
  const [selectedReason, setSelectedReason] = useState(null);

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const reasons = content.reasons || ['Too many emails', 'Content not relevant', 'Found better alternatives', 'Privacy concerns', 'Other'];
  const stayButton = content.stay_button || 'Stay Subscribed';
  const unsubscribeButton = content.unsubscribe_button || 'Unsubscribe';
  const continueUnsubLink = content.continue_unsubscribe_link || 'I still want to unsubscribe';

  const totalSteps = content.total_steps || (isDark ? 5 : 2);

  useEffect(() => {
    setCurrentStep(step, totalSteps);
  }, [step, setCurrentStep, totalSteps]);

  const goToStep = (newStep) => {
    trackClick(`step_${newStep}`, 'navigation', newStep === totalSteps);
    setStep(newStep);
  };

  const handleComplete = () => {
    trackClick('complete_unsubscribe', 'button', true);
    onAction('complete_unsubscribe');
  };

  const handleAbandon = () => {
    trackClick('cancel_button', 'button', false);
    onAction('abandon');
  };

  const handleStay = () => {
    trackClick('stay_subscribed', 'button', false);
    onAction('stay_subscribed');
  };

  // DARK PATTERN VERSION - Multi-step obstruction
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg max-w-lg w-full overflow-hidden">
          {/* Progress Bar */}
          <div className="bg-gray-50 px-6 py-4 border-b">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-500">Step {step} of {totalSteps}</span>
              <button
                onClick={handleAbandon}
                onMouseEnter={() => trackHover('cancel_button', 'button', false)}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-400 transition-all duration-300"
                style={{ width: `${(step / totalSteps) * 100}%` }}
              />
            </div>
          </div>

          {/* Step 1: Guilt Trip */}
          {step === 1 && (
            <div className="p-6 text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Heart className="w-8 h-8 text-red-500" />
              </div>
              <h2 className="text-xl font-bold text-gray-800 mb-3">We'll Miss You!</h2>
              <p className="text-gray-600 mb-6">
                Are you sure you want to unsubscribe? You'll miss out on exclusive deals,
                updates, and special content we've prepared just for you.
              </p>

              <button
                onClick={handleStay}
                onMouseEnter={() => trackHover('stay_subscribed', 'button', false)}
                className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold mb-3 hover:bg-blue-700 transition-colors"
              >
                {stayButton}
              </button>

              <button
                onClick={() => goToStep(2)}
                onMouseEnter={() => trackHover('continue_unsubscribe', 'button', true)}
                className="text-gray-400 text-sm hover:text-gray-500"
              >
                {continueUnsubLink}
              </button>
            </div>
          )}

          {/* Step 2: Select Reason */}
          {step === 2 && (
            <div className="p-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4">Help us understand why</h2>
              <p className="text-gray-600 text-sm mb-4">
                Please select a reason for leaving:
              </p>

              <div className="space-y-2 mb-6">
                {reasons.map((reason, index) => (
                  <label
                    key={index}
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedReason === reason
                        ? 'bg-blue-50 border-blue-300'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                    onClick={() => {
                      trackClick(`reason_${index}`, 'radio', false);
                      setSelectedReason(reason);
                    }}
                  >
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      selectedReason === reason ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                    }`}>
                      {selectedReason === reason && (
                        <div className="w-full h-full rounded-full bg-white scale-50" />
                      )}
                    </div>
                    <span className="text-gray-700">{reason}</span>
                  </label>
                ))}
              </div>

              <button
                onClick={() => goToStep(3)}
                onMouseEnter={() => trackHover('next_button', 'button', true)}
                disabled={!selectedReason}
                className="w-full bg-gray-800 text-white py-3 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Continue
              </button>
            </div>
          )}

          {/* Step 3: Confirm Email */}
          {step === 3 && (
            <div className="p-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4">Confirm Your Email</h2>
              <p className="text-gray-600 text-sm mb-4">
                Please enter your email address to confirm:
              </p>

              <input
                type="email"
                placeholder="your@email.com"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-gray-400"
                onFocus={() => trackClick('email_input', 'input', true)}
              />

              <button
                onClick={() => goToStep(4)}
                onMouseEnter={() => trackHover('verify_button', 'button', true)}
                className="w-full bg-gray-800 text-white py-3 rounded-xl font-semibold"
              >
                Verify
              </button>
            </div>
          )}

          {/* Step 4: Final Warning */}
          {step === 4 && (
            <div className="p-6 text-center">
              <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertTriangle className="w-8 h-8 text-yellow-600" />
              </div>
              <h2 className="text-xl font-bold text-gray-800 mb-3">Last Chance!</h2>
              <p className="text-gray-600 mb-6">
                This action cannot be undone. You will lose access to:
              </p>

              <ul className="text-left text-gray-600 mb-6 space-y-2">
                <li className="flex items-center gap-2">
                  <span className="text-red-500">✗</span> Exclusive member discounts
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-500">✗</span> Early access to sales
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-500">✗</span> Personalized recommendations
                </li>
              </ul>

              <button
                onClick={handleStay}
                onMouseEnter={() => trackHover('stay_subscribed_final', 'button', false)}
                className="w-full bg-green-600 text-white py-3 rounded-xl font-semibold mb-3 hover:bg-green-700 transition-colors"
              >
                Keep My Subscription
              </button>

              <button
                onClick={() => goToStep(5)}
                onMouseEnter={() => trackHover('final_unsubscribe', 'button', true)}
                className="text-gray-400 text-xs hover:text-gray-500"
              >
                Proceed to unsubscribe
              </button>
            </div>
          )}

          {/* Step 5: Final Confirmation (hidden button) */}
          {step === 5 && (
            <div className="p-6 text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Mail className="w-8 h-8 text-gray-400" />
              </div>
              <h2 className="text-lg font-bold text-gray-800 mb-3">Processing...</h2>
              <p className="text-gray-600 mb-6 text-sm">
                We're sad to see you go. Click below to complete your unsubscription.
              </p>

              {/* Hidden complete button - scroll needed */}
              <div className="h-32" />

              <button
                onClick={handleComplete}
                onMouseEnter={() => trackHover('complete_unsubscribe', 'button', true)}
                className="text-gray-300 text-xs hover:text-gray-400 mb-4"
              >
                Complete unsubscription
              </button>

              <p className="text-gray-400 text-xs">
                Changed your mind?{' '}
                <button
                  onClick={handleStay}
                  className="text-blue-500 hover:underline"
                >
                  Stay subscribed
                </button>
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION - One-click unsubscribe
  return (
    <div className="min-h-full bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-md w-full overflow-hidden">
        {step === 1 && (
          <div className="p-8 text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Mail className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-xl font-bold text-gray-800 mb-3">Unsubscribe</h2>
            <p className="text-gray-600 mb-6">
              Click below to unsubscribe from our newsletter.
            </p>

            <button
              onClick={handleComplete}
              onMouseEnter={() => trackHover('unsubscribe_button', 'button', true)}
              className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors mb-3"
            >
              {unsubscribeButton}
            </button>

            <button
              onClick={handleAbandon}
              onMouseEnter={() => trackHover('cancel_button', 'button', false)}
              className="text-gray-500 text-sm hover:text-gray-700"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
