/**
 * CookieConsent Task Component
 *
 * Dark Pattern Version (dp_01):
 * - Large "Accept All" button
 * - Small, muted "Manage Preferences" link
 * - Countdown timer creating urgency
 * - Preselected options
 *
 * Clean UI Version (clean_01):
 * - Equal-sized Accept/Reject buttons
 * - Clear options
 * - No manipulation
 */

import React, { useState } from 'react';
import { Cookie, Shield, Settings, ChevronDown, ChevronUp, X } from 'lucide-react';

export default function CookieConsent({ task, uiContent, onAction, trackClick, trackHover, trackExpand }) {
  const [showDetails, setShowDetails] = useState(false);
  const [cookies, setCookies] = useState({
    necessary: true,
    analytics: task.ui_type === 'dark',  // Preselected in dark version
    marketing: task.ui_type === 'dark',
    personalization: task.ui_type === 'dark',
  });

  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const siteName = content.site_name || 'TechNews Daily';
  const headline = content.headline || (isDark ? 'We Value Your Privacy' : 'Cookie Preferences');
  const acceptButtonText = content.accept_all_button || 'Accept All Cookies';
  const rejectButtonText = content.reject_all_button || 'Reject All';
  const manageButtonText = content.manage_button || 'Manage preferences';

  const handleToggle = (type) => {
    if (type === 'necessary') return; // Always required
    trackClick(`cookie_toggle_${type}`, 'checkbox', type === 'marketing' || type === 'analytics');
    setCookies(prev => ({ ...prev, [type]: !prev[type] }));
  };

  const handleAcceptAll = () => {
    trackClick('accept_all_button', 'button', false);
    onAction('accept_all');
  };

  const handleRejectAll = () => {
    trackClick('reject_all_button', 'button', true);
    onAction('reject_all');
  };

  const handleSavePreferences = () => {
    trackClick('save_preferences_button', 'button', true);
    const hasNonEssential = cookies.analytics || cookies.marketing || cookies.personalization;
    onAction(hasNonEssential ? 'accept_some' : 'reject_all');
  };

  const handleManageClick = () => {
    trackClick('manage_preferences_link', 'link', true);
    trackExpand();
    setShowDetails(true);
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-900/50 flex items-center justify-center p-4">
        {/* Website Preview Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-purple-100 opacity-50" />

        {/* Cookie Banner - Dark Pattern */}
        <div className="relative bg-white rounded-xl shadow-2xl max-w-lg w-full overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-4 text-white">
            <div className="flex items-center gap-3">
              <Cookie className="w-8 h-8" />
              <div>
                <h2 className="text-xl font-bold">{headline}</h2>
                <p className="text-purple-100 text-sm">This site uses cookies to enhance your experience</p>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            <p className="text-gray-600 text-sm mb-6">
              We use cookies and similar technologies to improve your browsing experience,
              personalize content and ads, provide social media features, and analyze our traffic.
              By clicking "Accept All", you consent to the use of ALL cookies.
            </p>

            {/* Main Action - Large Accept Button */}
            <button
              onClick={handleAcceptAll}
              onMouseEnter={() => trackHover('accept_all_button', 'button', false)}
              className="w-full bg-gradient-to-r from-green-500 to-green-600 text-white py-4 rounded-xl text-lg font-bold shadow-lg hover:shadow-xl transition-all hover:scale-[1.02] mb-3"
            >
              {acceptButtonText}
            </button>

            {/* Secondary - Very Small Manage Link */}
            {!showDetails && (
              <div className="text-center">
                <button
                  onClick={handleManageClick}
                  onMouseEnter={() => trackHover('manage_preferences_link', 'link', true)}
                  className="text-xs text-gray-400 hover:text-gray-500 underline"
                >
                  Manage preferences
                </button>
              </div>
            )}

            {/* Expanded Details (Hidden by default) */}
            {showDetails && (
              <div className="mt-4 border-t pt-4">
                <div className="space-y-3">
                  {/* Necessary Cookies */}
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-800">Necessary</p>
                      <p className="text-xs text-gray-500">Required for the website to function</p>
                    </div>
                    <div className="w-12 h-6 bg-green-500 rounded-full relative cursor-not-allowed">
                      <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full" />
                    </div>
                  </div>

                  {/* Analytics */}
                  <div
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer"
                    onClick={() => handleToggle('analytics')}
                    onMouseEnter={() => trackHover('analytics_toggle', 'checkbox', true)}
                  >
                    <div>
                      <p className="font-medium text-gray-800">Analytics</p>
                      <p className="text-xs text-gray-500">Help us improve our website</p>
                    </div>
                    <div className={`w-12 h-6 rounded-full relative transition-colors ${cookies.analytics ? 'bg-green-500' : 'bg-gray-300'}`}>
                      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${cookies.analytics ? 'right-1' : 'left-1'}`} />
                    </div>
                  </div>

                  {/* Marketing - With confusing label */}
                  <div
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer"
                    onClick={() => handleToggle('marketing')}
                    onMouseEnter={() => trackHover('marketing_toggle', 'checkbox', true)}
                  >
                    <div>
                      <p className="font-medium text-gray-800">Partners & Affiliates</p>
                      <p className="text-xs text-gray-400">Don't disable to keep receiving relevant offers</p>
                    </div>
                    <div className={`w-12 h-6 rounded-full relative transition-colors ${cookies.marketing ? 'bg-green-500' : 'bg-gray-300'}`}>
                      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${cookies.marketing ? 'right-1' : 'left-1'}`} />
                    </div>
                  </div>

                  {/* Personalization */}
                  <div
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer"
                    onClick={() => handleToggle('personalization')}
                    onMouseEnter={() => trackHover('personalization_toggle', 'checkbox', true)}
                  >
                    <div>
                      <p className="font-medium text-gray-800">Personalization</p>
                      <p className="text-xs text-gray-500">Customize your experience</p>
                    </div>
                    <div className={`w-12 h-6 rounded-full relative transition-colors ${cookies.personalization ? 'bg-green-500' : 'bg-gray-300'}`}>
                      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${cookies.personalization ? 'right-1' : 'left-1'}`} />
                    </div>
                  </div>
                </div>

                {/* Save button (small, less prominent) */}
                <button
                  onClick={handleSavePreferences}
                  onMouseEnter={() => trackHover('save_preferences_button', 'button', true)}
                  className="w-full mt-4 py-2 border border-gray-300 text-gray-600 rounded-lg text-sm hover:bg-gray-50"
                >
                  Save preferences
                </button>

                {/* Hidden reject all - tiny text */}
                <p
                  onClick={handleRejectAll}
                  onMouseEnter={() => trackHover('reject_all_link', 'link', true)}
                  className="text-center mt-2 text-[10px] text-gray-300 cursor-pointer hover:text-gray-400"
                >
                  Reject all non-essential
                </p>
              </div>
            )}
          </div>

          {/* Fine Print */}
          <div className="px-6 pb-4">
            <p
              className="text-[9px] text-gray-300 leading-tight"
              onMouseEnter={() => trackHover('fine_print', 'fine_print', false)}
            >
              By continuing to use this site, you agree to our privacy policy and terms of service.
              Cookie preferences can be changed at any time in your browser settings.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-900/50 flex items-center justify-center p-4">
      {/* Website Preview Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-purple-100 opacity-50" />

      {/* Cookie Banner - Clean UI */}
      <div className="relative bg-white rounded-xl shadow-xl max-w-lg w-full overflow-hidden">
        {/* Header */}
        <div className="bg-gray-100 px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <Cookie className="w-6 h-6 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-800">{headline}</h2>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          <p className="text-gray-600 text-sm mb-6">
            We use cookies to improve your experience. You can accept all cookies or manage your preferences.
          </p>

          {/* Equal-sized buttons */}
          <div className="flex gap-3 mb-4">
            <button
              onClick={handleRejectAll}
              onMouseEnter={() => trackHover('reject_all_button', 'button', true)}
              className="flex-1 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              {rejectButtonText}
            </button>
            <button
              onClick={handleAcceptAll}
              onMouseEnter={() => trackHover('accept_all_button', 'button', false)}
              className="flex-1 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              {content.accept_all_button || 'Accept All'}
            </button>
          </div>

          {/* Manage Preferences - Clearly visible */}
          <button
            onClick={() => {
              trackClick('manage_preferences_button', 'button', true);
              trackExpand();
              setShowDetails(!showDetails);
            }}
            onMouseEnter={() => trackHover('manage_preferences_button', 'button', true)}
            className="w-full py-2 text-blue-600 font-medium flex items-center justify-center gap-2 hover:bg-blue-50 rounded-lg transition-colors"
          >
            <Settings className="w-4 h-4" />
            Manage Preferences
            {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {/* Expanded Details */}
          {showDetails && (
            <div className="mt-4 border-t pt-4">
              <div className="space-y-3">
                {/* Necessary Cookies */}
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-800">Necessary Cookies</p>
                    <p className="text-xs text-gray-500">Required for the website to work properly</p>
                  </div>
                  <span className="text-sm text-gray-500">Always on</span>
                </div>

                {/* Analytics - OFF by default */}
                <div
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer"
                  onClick={() => handleToggle('analytics')}
                  onMouseEnter={() => trackHover('analytics_toggle', 'checkbox', true)}
                >
                  <div>
                    <p className="font-medium text-gray-800">Analytics Cookies</p>
                    <p className="text-xs text-gray-500">Help us understand how visitors use our site</p>
                  </div>
                  <div className={`w-12 h-6 rounded-full relative transition-colors ${cookies.analytics ? 'bg-blue-500' : 'bg-gray-300'}`}>
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${cookies.analytics ? 'right-1' : 'left-1'}`} />
                  </div>
                </div>

                {/* Marketing - OFF by default */}
                <div
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer"
                  onClick={() => handleToggle('marketing')}
                  onMouseEnter={() => trackHover('marketing_toggle', 'checkbox', true)}
                >
                  <div>
                    <p className="font-medium text-gray-800">Marketing Cookies</p>
                    <p className="text-xs text-gray-500">Used to show you relevant ads</p>
                  </div>
                  <div className={`w-12 h-6 rounded-full relative transition-colors ${cookies.marketing ? 'bg-blue-500' : 'bg-gray-300'}`}>
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${cookies.marketing ? 'right-1' : 'left-1'}`} />
                  </div>
                </div>
              </div>

              {/* Save Preferences */}
              <button
                onClick={handleSavePreferences}
                onMouseEnter={() => trackHover('save_preferences_button', 'button', true)}
                className="w-full mt-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
              >
                Save Preferences
              </button>
            </div>
          )}
        </div>

        {/* Clear Privacy Link */}
        <div className="px-6 pb-4 text-center">
          <a
            href="#"
            className="text-sm text-blue-600 hover:underline"
            onClick={(e) => {
              e.preventDefault();
              trackClick('privacy_policy_link', 'link', false);
            }}
          >
            Learn more about our privacy policy
          </a>
        </div>
      </div>
    </div>
  );
}
