/**
 * PrivacySettings Task Component
 *
 * Dark Pattern Version (dp_07): Confusing double-negatives, sharing ON by default
 * Clean UI Version (clean_07): Clear toggles, sharing OFF by default
 */

import React, { useState } from 'react';
import { Shield, Eye, Share2, Bell, Lock, Check } from 'lucide-react';

export default function PrivacySettings({ task, uiContent, onAction, trackClick, trackHover }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const saveButton = content.save_button || 'Save Preferences';

  // Dark version: sharing ON by default with confusing labels
  // Clean version: sharing OFF by default with clear labels
  const [settings, setSettings] = useState({
    dataSharing: isDark,
    personalizedAds: isDark,
    thirdPartyAccess: isDark,
    analytics: isDark,
  });

  const handleToggle = (key) => {
    trackClick(`toggle_${key}`, 'toggle', key === 'dataSharing');
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = () => {
    trackClick('save_button', 'button', !settings.dataSharing);
    const sharingEnabled = settings.dataSharing || settings.thirdPartyAccess;
    onAction(sharingEnabled ? 'leave_enabled' : 'disable_sharing');
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-100 p-4 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg max-w-lg w-full">
          <div className="px-6 py-4 border-b">
            <h2 className="text-xl font-bold text-gray-800">Privacy Preferences</h2>
            <p className="text-gray-500 text-sm">Manage how your data is used</p>
          </div>

          <div className="p-6 space-y-4">
            {/* Confusing double-negative: "Don't disable" */}
            <div
              className="flex items-center justify-between p-4 bg-gray-50 rounded-xl cursor-pointer"
              onClick={() => handleToggle('dataSharing')}
              onMouseEnter={() => trackHover('dataSharing_toggle', 'toggle', true)}
            >
              <div className="flex items-center gap-3">
                <Share2 className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-800">Data Sharing Preferences</p>
                  <p className="text-xs text-gray-400">
                    Don't disable to keep receiving relevant content
                  </p>
                </div>
              </div>
              <div className={`w-12 h-6 rounded-full relative transition-colors ${
                settings.dataSharing ? 'bg-green-500' : 'bg-gray-300'
              }`}>
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                  settings.dataSharing ? 'right-1' : 'left-1'
                }`} />
              </div>
            </div>

            {/* Another confusing one */}
            <div
              className="flex items-center justify-between p-4 bg-gray-50 rounded-xl cursor-pointer"
              onClick={() => handleToggle('personalizedAds')}
              onMouseEnter={() => trackHover('personalizedAds_toggle', 'toggle', true)}
            >
              <div className="flex items-center gap-3">
                <Eye className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-800">Personalized Experience</p>
                  <p className="text-xs text-gray-400">
                    Opt out of non-personalized ads
                  </p>
                </div>
              </div>
              <div className={`w-12 h-6 rounded-full relative transition-colors ${
                settings.personalizedAds ? 'bg-green-500' : 'bg-gray-300'
              }`}>
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                  settings.personalizedAds ? 'right-1' : 'left-1'
                }`} />
              </div>
            </div>

            {/* Hidden in scroll */}
            <div
              className="flex items-center justify-between p-4 bg-gray-50 rounded-xl cursor-pointer"
              onClick={() => handleToggle('thirdPartyAccess')}
              onMouseEnter={() => trackHover('thirdPartyAccess_toggle', 'toggle', true)}
            >
              <div className="flex items-center gap-3">
                <Lock className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-800">Partner Access</p>
                  <p className="text-xs text-gray-400">
                    Allow trusted partners to enhance your experience
                  </p>
                </div>
              </div>
              <div className={`w-12 h-6 rounded-full relative transition-colors ${
                settings.thirdPartyAccess ? 'bg-green-500' : 'bg-gray-300'
              }`}>
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                  settings.thirdPartyAccess ? 'right-1' : 'left-1'
                }`} />
              </div>
            </div>

            <div
              className="flex items-center justify-between p-4 bg-gray-50 rounded-xl cursor-pointer"
              onClick={() => handleToggle('analytics')}
              onMouseEnter={() => trackHover('analytics_toggle', 'toggle', false)}
            >
              <div className="flex items-center gap-3">
                <Bell className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-800">Usage Analytics</p>
                  <p className="text-xs text-gray-400">
                    Help us improve by sharing usage data
                  </p>
                </div>
              </div>
              <div className={`w-12 h-6 rounded-full relative transition-colors ${
                settings.analytics ? 'bg-green-500' : 'bg-gray-300'
              }`}>
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                  settings.analytics ? 'right-1' : 'left-1'
                }`} />
              </div>
            </div>

            {/* Fine print */}
            <p
              className="text-[10px] text-gray-300 mt-4"
              onMouseEnter={() => trackHover('fine_print', 'fine_print', false)}
            >
              Note: Disabling these options may affect your experience. Some features require data sharing.
            </p>
          </div>

          <div className="px-6 pb-6">
            <button
              onClick={handleSave}
              onMouseEnter={() => trackHover('save_button', 'button', !settings.dataSharing)}
              className="w-full bg-gray-800 text-white py-3 rounded-xl font-semibold"
            >
              Save Preferences
            </button>
          </div>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-100 p-4 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-lg max-w-lg w-full">
        <div className="px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-blue-600" />
            <div>
              <h2 className="text-xl font-bold text-gray-800">Privacy Settings</h2>
              <p className="text-gray-500 text-sm">Control how your data is used</p>
            </div>
          </div>
        </div>

        <div className="p-6 space-y-4">
          {/* Clear labels - OFF by default */}
          <div
            className="flex items-center justify-between p-4 border rounded-xl cursor-pointer hover:bg-gray-50"
            onClick={() => handleToggle('dataSharing')}
            onMouseEnter={() => trackHover('dataSharing_toggle', 'toggle', true)}
          >
            <div className="flex items-center gap-3">
              <Share2 className="w-5 h-5 text-gray-500" />
              <div>
                <p className="font-medium text-gray-800">Share Data with Third Parties</p>
                <p className="text-xs text-gray-500">
                  Allow us to share your data with partners for advertising
                </p>
              </div>
            </div>
            <div className={`w-12 h-6 rounded-full relative transition-colors ${
              settings.dataSharing ? 'bg-blue-500' : 'bg-gray-300'
            }`}>
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                settings.dataSharing ? 'right-1' : 'left-1'
              }`} />
            </div>
          </div>

          <div
            className="flex items-center justify-between p-4 border rounded-xl cursor-pointer hover:bg-gray-50"
            onClick={() => handleToggle('personalizedAds')}
            onMouseEnter={() => trackHover('personalizedAds_toggle', 'toggle', true)}
          >
            <div className="flex items-center gap-3">
              <Eye className="w-5 h-5 text-gray-500" />
              <div>
                <p className="font-medium text-gray-800">Personalized Advertisements</p>
                <p className="text-xs text-gray-500">
                  Show ads based on your activity
                </p>
              </div>
            </div>
            <div className={`w-12 h-6 rounded-full relative transition-colors ${
              settings.personalizedAds ? 'bg-blue-500' : 'bg-gray-300'
            }`}>
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                settings.personalizedAds ? 'right-1' : 'left-1'
              }`} />
            </div>
          </div>

          <div
            className="flex items-center justify-between p-4 border rounded-xl cursor-pointer hover:bg-gray-50"
            onClick={() => handleToggle('analytics')}
            onMouseEnter={() => trackHover('analytics_toggle', 'toggle', false)}
          >
            <div className="flex items-center gap-3">
              <Bell className="w-5 h-5 text-gray-500" />
              <div>
                <p className="font-medium text-gray-800">Usage Analytics</p>
                <p className="text-xs text-gray-500">
                  Help us improve by sharing anonymous usage data
                </p>
              </div>
            </div>
            <div className={`w-12 h-6 rounded-full relative transition-colors ${
              settings.analytics ? 'bg-blue-500' : 'bg-gray-300'
            }`}>
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                settings.analytics ? 'right-1' : 'left-1'
              }`} />
            </div>
          </div>
        </div>

        <div className="px-6 pb-6">
          <button
            onClick={handleSave}
            onMouseEnter={() => trackHover('save_button', 'button', !settings.dataSharing)}
            className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
