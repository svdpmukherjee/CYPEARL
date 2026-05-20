/**
 * DeclineOffer Task Component
 *
 * Dark Pattern Version (dp_08):
 * - Confirmshaming: "No thanks, I don't like saving money"
 *
 * Clean UI Version (clean_08):
 * - Neutral "No thanks" decline option
 */

import React from 'react';
import { Gift, Percent, X, Sparkles } from 'lucide-react';

export default function DeclineOffer({ task, uiContent, onAction, trackClick, trackHover }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const headline = content.headline || (isDark ? "Wait! Don't Leave!" : 'Special Offer');
  const offer = content.offer || 'Get 25% OFF your first order!';
  const acceptButton = content.accept_button || (isDark ? 'Yes! I Want 25% Off' : 'Sign Up');
  const declineButton = content.decline_button || (isDark ? 'No thanks, I prefer paying full price' : 'No thanks');

  const handleAccept = () => {
    trackClick('accept_offer_button', 'button', false);
    onAction('accept');
  };

  const handleDecline = () => {
    trackClick('decline_offer_button', 'button', true);
    onAction('decline');
  };

  const handleClose = () => {
    trackClick('close_button', 'button', true);
    onAction('decline');
  };

  // DARK PATTERN VERSION - Confirmshaming
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-900/70 flex items-center justify-center p-4">
        {/* Website Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-indigo-100 opacity-30" />

        {/* Popup */}
        <div className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden animate-bounce-once">
          {/* Close button - very small */}
          <button
            onClick={handleClose}
            onMouseEnter={() => trackHover('close_button', 'button', true)}
            className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center text-gray-300 hover:text-gray-400 z-10"
          >
            <X className="w-4 h-4" />
          </button>

          {/* Header Image */}
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 py-8 text-center relative overflow-hidden">
            <Sparkles className="absolute top-2 left-4 w-6 h-6 text-white/30" />
            <Sparkles className="absolute bottom-2 right-4 w-6 h-6 text-white/30" />
            <div className="relative">
              <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-3">
                <Gift className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">{headline}</h2>
            </div>
          </div>

          {/* Content */}
          <div className="px-6 py-6 text-center">
            <div className="inline-block bg-yellow-100 text-yellow-800 px-4 py-1 rounded-full text-sm font-medium mb-4">
              🎉 EXCLUSIVE OFFER
            </div>

            <h3 className="text-xl font-bold text-gray-800 mb-2">
              {offer}
            </h3>

            <p className="text-gray-600 mb-6">
              Sign up for our newsletter and receive an exclusive discount code
              delivered straight to your inbox.
            </p>

            {/* Email Input */}
            <input
              type="email"
              placeholder="Enter your email"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-purple-500"
              onFocus={() => trackClick('email_input', 'input', false)}
            />

            {/* Accept Button - Large and prominent */}
            <button
              onClick={handleAccept}
              onMouseEnter={() => trackHover('accept_offer_button', 'button', false)}
              className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-4 rounded-xl font-bold text-lg shadow-lg hover:shadow-xl transition-all hover:scale-[1.02] mb-4"
            >
              {acceptButton}
            </button>

            {/* Decline - Confirmshaming */}
            <button
              onClick={handleDecline}
              onMouseEnter={() => trackHover('decline_offer_button', 'button', true)}
              className="text-gray-400 text-sm hover:text-gray-500 italic"
            >
              {declineButton}
            </button>
          </div>

          {/* Fine Print */}
          <div className="px-6 pb-4">
            <p
              className="text-[9px] text-gray-300 text-center"
              onMouseEnter={() => trackHover('fine_print', 'fine_print', false)}
            >
              By signing up, you agree to receive marketing emails. Unsubscribe anytime.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-900/70 flex items-center justify-center p-4">
      {/* Website Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-indigo-100 opacity-30" />

      {/* Popup */}
      <div className="relative bg-white rounded-2xl shadow-xl max-w-md w-full overflow-hidden">
        {/* Close button - clearly visible */}
        <button
          onClick={handleClose}
          onMouseEnter={() => trackHover('close_button', 'button', true)}
          className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full z-10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="bg-blue-50 py-6 text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <Percent className="w-8 h-8 text-blue-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-800">{headline}</h2>
        </div>

        {/* Content */}
        <div className="px-6 py-6 text-center">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">
            Get 25% Off Your First Order
          </h3>

          <p className="text-gray-600 mb-6 text-sm">
            Sign up for our newsletter to receive your discount code.
          </p>

          {/* Email Input */}
          <input
            type="email"
            placeholder="Your email address"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
            onFocus={() => trackClick('email_input', 'input', false)}
          />

          {/* Buttons - Equal treatment */}
          <div className="flex gap-3">
            <button
              onClick={handleDecline}
              onMouseEnter={() => trackHover('decline_offer_button', 'button', true)}
              className="flex-1 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              {declineButton}
            </button>
            <button
              onClick={handleAccept}
              onMouseEnter={() => trackHover('accept_offer_button', 'button', false)}
              className="flex-1 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              {acceptButton}
            </button>
          </div>
        </div>

        {/* Privacy Note */}
        <div className="px-6 pb-4">
          <p className="text-xs text-gray-500 text-center">
            We respect your privacy. Unsubscribe at any time.
          </p>
        </div>
      </div>
    </div>
  );
}
