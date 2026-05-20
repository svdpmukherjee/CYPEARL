/**
 * AccountDeletion Task Component
 *
 * Dark Pattern Version (dp_04): Multi-step obstruction
 * Clean UI Version (clean_04): 2-step simple deletion
 */

import React, { useState, useEffect } from 'react';
import { UserX, AlertTriangle, Check, ChevronRight, Heart } from 'lucide-react';

export default function AccountDeletion({ task, uiContent, onAction, trackClick, trackHover, setCurrentStep }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const serviceName = content.service_name || 'CloudSync Plus';
  const confirmSteps = content.confirmation_steps || (isDark ? 5 : 2);

  const [step, setStep] = useState(1);
  const totalSteps = confirmSteps;

  useEffect(() => {
    setCurrentStep(step, totalSteps);
  }, [step, setCurrentStep, totalSteps]);

  const handleComplete = () => {
    trackClick('delete_account_button', 'button', true);
    onAction('complete_deletion');
  };

  const handleAbandon = () => {
    trackClick('cancel_button', 'button', false);
    onAction('abandon');
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg max-w-md w-full">
          {/* Progress */}
          <div className="px-6 py-4 border-b">
            <div className="flex justify-between text-sm text-gray-500 mb-2">
              <span>Delete Account</span>
              <span>Step {step} of {totalSteps}</span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full">
              <div
                className="h-full bg-red-500 rounded-full transition-all"
                style={{ width: `${(step / totalSteps) * 100}%` }}
              />
            </div>
          </div>

          {/* Step 1: Guilt trip */}
          {step === 1 && (
            <div className="p-6 text-center">
              <Heart className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <h2 className="text-xl font-bold mb-3">We'll Be Sad to See You Go</h2>
              <p className="text-gray-600 mb-6">
                You've been with us for 2 years! Are you sure you want to leave?
              </p>
              <button
                onClick={handleAbandon}
                onMouseEnter={() => trackHover('stay_button', 'button', false)}
                className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold mb-3"
              >
                Keep My Account
              </button>
              <button
                onClick={() => { trackClick('next', 'button', true); setStep(2); }}
                className="text-gray-400 text-sm"
              >
                Continue to delete
              </button>
            </div>
          )}

          {/* Step 2: Reason */}
          {step === 2 && (
            <div className="p-6">
              <h2 className="text-lg font-bold mb-4">Why are you leaving?</h2>
              <div className="space-y-2 mb-6">
                {['Privacy concerns', 'Found alternative', 'Too expensive', 'Other'].map((r, i) => (
                  <button
                    key={i}
                    onClick={() => { trackClick(`reason_${i}`, 'radio', false); setStep(3); }}
                    className="w-full p-3 border rounded-lg text-left hover:bg-gray-50"
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 3: Enter password */}
          {step === 3 && (
            <div className="p-6">
              <h2 className="text-lg font-bold mb-4">Confirm Your Identity</h2>
              <input
                type="password"
                placeholder="Enter your password"
                className="w-full px-4 py-3 border rounded-lg mb-4"
                onFocus={() => trackClick('password_input', 'input', true)}
              />
              <button
                onClick={() => { trackClick('verify', 'button', true); setStep(4); }}
                className="w-full bg-gray-800 text-white py-3 rounded-xl font-semibold"
              >
                Verify
              </button>
            </div>
          )}

          {/* Step 4: Warning */}
          {step === 4 && (
            <div className="p-6 text-center">
              <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
              <h2 className="text-xl font-bold mb-3">Final Warning!</h2>
              <p className="text-gray-600 mb-6">This will permanently delete all your data.</p>
              <button
                onClick={handleAbandon}
                onMouseEnter={() => trackHover('cancel_final', 'button', false)}
                className="w-full bg-green-600 text-white py-3 rounded-xl font-semibold mb-3"
              >
                Cancel
              </button>
              <button
                onClick={() => { trackClick('proceed_final', 'button', true); setStep(5); }}
                className="text-gray-400 text-xs"
              >
                Proceed anyway
              </button>
            </div>
          )}

          {/* Step 5: Final (hidden button) */}
          {step === 5 && (
            <div className="p-6 text-center">
              <UserX className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h2 className="text-lg font-bold mb-6">Almost done...</h2>
              <div className="h-24" />
              <button
                onClick={handleComplete}
                onMouseEnter={() => trackHover('final_delete', 'button', true)}
                className="text-gray-300 text-xs hover:text-gray-400"
              >
                Permanently delete my account
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-md w-full p-6">
        {step === 1 && (
          <div className="text-center">
            <UserX className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-3">Delete Account</h2>
            <p className="text-gray-600 mb-6">
              Are you sure? This will permanently delete your account and all data.
            </p>
            <button
              onClick={() => { trackClick('confirm_step', 'button', true); setStep(2); }}
              onMouseEnter={() => trackHover('delete_button', 'button', true)}
              className="w-full bg-red-600 text-white py-3 rounded-xl font-semibold mb-3"
            >
              Yes, Delete My Account
            </button>
            <button
              onClick={handleAbandon}
              className="w-full text-gray-500"
            >
              Cancel
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="text-center">
            <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-3">Confirm Deletion</h2>
            <p className="text-gray-600 mb-6">Type "DELETE" to confirm:</p>
            <input
              type="text"
              placeholder="Type DELETE"
              className="w-full px-4 py-3 border rounded-lg mb-4 text-center"
              onFocus={() => trackClick('confirm_input', 'input', true)}
            />
            <button
              onClick={handleComplete}
              onMouseEnter={() => trackHover('final_delete', 'button', true)}
              className="w-full bg-red-600 text-white py-3 rounded-xl font-semibold"
            >
              Permanently Delete
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
