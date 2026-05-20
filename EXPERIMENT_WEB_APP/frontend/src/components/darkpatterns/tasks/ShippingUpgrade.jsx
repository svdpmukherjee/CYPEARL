/**
 * ShippingUpgrade Task Component
 *
 * Dark Pattern Version (dp_06): Premium preselected, standard hidden
 * Clean UI Version (clean_06): Standard as default, premium clearly optional
 */

import React, { useState } from 'react';
import { Truck, Zap, Clock, Check } from 'lucide-react';

export default function ShippingUpgrade({ task, uiContent, onAction, trackClick, trackHover }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const continueButton = content.continue_button || 'Continue to Payment';

  const [selected, setSelected] = useState(isDark ? 'premium' : 'standard');

  const handleContinue = () => {
    trackClick('continue_button', 'button', selected === 'standard');
    onAction(selected === 'standard' ? 'select_standard' : 'keep_premium');
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-100 p-4 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg max-w-md w-full p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-2">Shipping Options</h2>
          <p className="text-gray-500 text-sm mb-6">Choose your delivery speed</p>

          {/* Premium - Preselected and Prominent */}
          <div
            onClick={() => { trackClick('premium_option', 'radio', false); setSelected('premium'); }}
            onMouseEnter={() => trackHover('premium_option', 'option', false)}
            className={`p-4 rounded-xl mb-3 cursor-pointer transition-all ${
              selected === 'premium'
                ? 'bg-green-50 border-2 border-green-500 shadow-md'
                : 'border border-gray-200'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Zap className="w-6 h-6 text-green-600" />
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-gray-800">Express Delivery</span>
                    <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full">
                      FASTEST
                    </span>
                  </div>
                  <p className="text-sm text-gray-500">1-2 business days</p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-bold text-gray-800">$14.99</p>
                <div className={`w-5 h-5 rounded-full border-2 ml-auto mt-1 ${
                  selected === 'premium' ? 'bg-green-500 border-green-500' : 'border-gray-300'
                }`}>
                  {selected === 'premium' && <Check className="w-4 h-4 text-white" />}
                </div>
              </div>
            </div>
          </div>

          {/* Standard - Less prominent */}
          <div
            onClick={() => { trackClick('standard_option', 'radio', true); setSelected('standard'); }}
            onMouseEnter={() => trackHover('standard_option', 'option', true)}
            className={`p-4 rounded-xl mb-6 cursor-pointer transition-all ${
              selected === 'standard'
                ? 'bg-gray-50 border-2 border-gray-400'
                : 'border border-gray-100'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Truck className="w-6 h-6 text-gray-400" />
                <div>
                  <span className="text-gray-600">Standard</span>
                  <p className="text-sm text-gray-400">5-7 business days</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-gray-500">Free</p>
                <div className={`w-5 h-5 rounded-full border-2 ml-auto mt-1 ${
                  selected === 'standard' ? 'bg-gray-400 border-gray-400' : 'border-gray-200'
                }`}>
                  {selected === 'standard' && <Check className="w-4 h-4 text-white" />}
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={handleContinue}
            onMouseEnter={() => trackHover('continue_button', 'button', selected === 'standard')}
            className="w-full bg-gray-800 text-white py-3 rounded-xl font-semibold"
          >
            Continue with {selected === 'premium' ? 'Express' : 'Standard'} Shipping
          </button>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-100 p-4 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-lg max-w-md w-full p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-2">Choose Shipping</h2>
        <p className="text-gray-500 text-sm mb-6">Select your preferred delivery option</p>

        {/* Standard - Default */}
        <div
          onClick={() => { trackClick('standard_option', 'radio', true); setSelected('standard'); }}
          onMouseEnter={() => trackHover('standard_option', 'option', true)}
          className={`p-4 rounded-xl mb-3 cursor-pointer transition-all ${
            selected === 'standard'
              ? 'bg-blue-50 border-2 border-blue-500'
              : 'border border-gray-200 hover:border-gray-300'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Truck className="w-6 h-6 text-blue-600" />
              <div>
                <span className="font-medium text-gray-800">Standard Shipping</span>
                <p className="text-sm text-gray-500">5-7 business days</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-medium text-green-600">Free</p>
              <div className={`w-5 h-5 rounded-full border-2 ml-auto mt-1 ${
                selected === 'standard' ? 'bg-blue-500 border-blue-500' : 'border-gray-300'
              }`}>
                {selected === 'standard' && <Check className="w-4 h-4 text-white" />}
              </div>
            </div>
          </div>
        </div>

        {/* Express - Optional upgrade */}
        <div
          onClick={() => { trackClick('premium_option', 'radio', false); setSelected('premium'); }}
          onMouseEnter={() => trackHover('premium_option', 'option', false)}
          className={`p-4 rounded-xl mb-6 cursor-pointer transition-all ${
            selected === 'premium'
              ? 'bg-blue-50 border-2 border-blue-500'
              : 'border border-gray-200 hover:border-gray-300'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-orange-500" />
              <div>
                <span className="font-medium text-gray-800">Express Shipping</span>
                <p className="text-sm text-gray-500">1-2 business days</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-medium text-gray-800">$14.99</p>
              <div className={`w-5 h-5 rounded-full border-2 ml-auto mt-1 ${
                selected === 'premium' ? 'bg-blue-500 border-blue-500' : 'border-gray-300'
              }`}>
                {selected === 'premium' && <Check className="w-4 h-4 text-white" />}
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleContinue}
          onMouseEnter={() => trackHover('continue_button', 'button', selected === 'standard')}
          className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold"
        >
          Continue
        </button>
      </div>
    </div>
  );
}
