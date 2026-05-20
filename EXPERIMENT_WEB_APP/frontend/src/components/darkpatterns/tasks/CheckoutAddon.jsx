/**
 * CheckoutAddon Task Component
 *
 * Dark Pattern Version (dp_05):
 * - Preselected insurance addon
 * - Muted uncheck option
 * - Time pressure with countdown
 *
 * Clean UI Version (clean_05):
 * - Addon unchecked by default
 * - Clear opt-in choice
 */

import React, { useState } from 'react';
import { ShoppingCart, Shield, Truck, CreditCard, Check } from 'lucide-react';

export default function CheckoutAddon({ task, uiContent, onAction, trackClick, trackHover, trackExpand }) {
  const isDark = task.ui_type === 'dark';

  // Use content from backend with fallbacks
  const content = uiContent || {};
  const storeName = content.store_name || 'ShopEase';
  const mainProduct = content.main_product || { name: 'Premium Wireless Headphones', price: 79.99 };
  const addon = content.addon || { name: 'Protection Plus', price: 12.99, description: '2-year protection' };
  const checkoutButtonText = content.checkout_button || 'Place Order';

  const basePrice = mainProduct.price;
  const addonPrice = addon.price;

  // Preselected in dark version (or if addon.preselected is true)
  const [hasAddon, setHasAddon] = useState(addon.preselected !== undefined ? addon.preselected : isDark);

  const total = hasAddon ? basePrice + addonPrice : basePrice;

  const handleToggleAddon = () => {
    trackClick('addon_checkbox', 'checkbox', true);
    setHasAddon(!hasAddon);
  };

  const handleProceed = () => {
    trackClick('proceed_button', 'button', !hasAddon);
    onAction(hasAddon ? 'keep_addon' : 'proceed_without_addon');
  };

  // DARK PATTERN VERSION
  if (isDark) {
    return (
      <div className="min-h-full bg-gray-100 p-4 md:p-8">
        <div className="max-w-2xl mx-auto">
          {/* Store Header */}
          <div className="bg-white rounded-t-xl px-6 py-4 border-b flex items-center gap-3">
            <ShoppingCart className="w-6 h-6 text-orange-500" />
            <span className="font-bold text-xl text-gray-800">{storeName} Checkout</span>
          </div>

          {/* Main Content */}
          <div className="bg-white px-6 py-6">
            {/* Order Summary */}
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Order Summary</h2>

            {/* Product */}
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg mb-4">
              <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center">
                <span className="text-2xl">📦</span>
              </div>
              <div className="flex-1">
                <p className="font-medium text-gray-800">{mainProduct.name}</p>
                <p className="text-sm text-gray-500">Color: Black | Qty: 1</p>
              </div>
              <p className="font-semibold text-gray-800">${basePrice.toFixed(2)}</p>
            </div>

            {/* Protection Plan - DARK PATTERN */}
            <div
              className={`p-4 rounded-lg mb-4 cursor-pointer transition-colors ${
                hasAddon
                  ? 'bg-green-50 border-2 border-green-500'
                  : 'bg-gray-50 border border-gray-200'
              }`}
              onClick={handleToggleAddon}
              onMouseEnter={() => trackHover('addon_section', 'checkbox', true)}
            >
              <div className="flex items-start gap-3">
                {/* Checkbox - styled to blend in when unchecking */}
                <div className={`mt-1 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                  hasAddon
                    ? 'bg-green-500 border-green-500'
                    : 'border-gray-300'
                }`}>
                  {hasAddon && <Check className="w-3 h-3 text-white" />}
                </div>

                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-green-600" />
                    <span className="font-semibold text-gray-800">Protection Plus</span>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Recommended</span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    2-year protection against drops, spills, and defects. Includes free repairs and replacements.
                  </p>
                  <p className="text-sm font-medium text-green-600 mt-1">Only ${addonPrice.toFixed(2)}</p>

                  {/* Fine print - very small */}
                  <p
                    className="text-[10px] text-gray-400 mt-2"
                    onMouseEnter={() => trackHover('addon_fine_print', 'fine_print', false)}
                  >
                    Auto-renews annually. Cancel anytime.
                  </p>
                </div>
              </div>

              {/* Uncheck instruction - barely visible */}
              {hasAddon && (
                <p className="text-[10px] text-gray-300 mt-2 text-right">
                  Click to remove protection
                </p>
              )}
            </div>

            {/* Shipping */}
            <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg mb-6">
              <Truck className="w-5 h-5 text-gray-400" />
              <div className="flex-1">
                <p className="text-sm text-gray-600">Standard Shipping</p>
                <p className="text-xs text-gray-400">Arrives in 5-7 business days</p>
              </div>
              <p className="text-sm text-green-600 font-medium">FREE</p>
            </div>

            {/* Order Total */}
            <div className="border-t pt-4">
              <div className="flex justify-between mb-2">
                <span className="text-gray-600">Subtotal</span>
                <span className="text-gray-800">${basePrice.toFixed(2)}</span>
              </div>
              {hasAddon && (
                <div className="flex justify-between mb-2">
                  <span className="text-gray-600">Protection Plus</span>
                  <span className="text-gray-800">${addonPrice.toFixed(2)}</span>
                </div>
              )}
              <div className="flex justify-between mb-2">
                <span className="text-gray-600">Shipping</span>
                <span className="text-green-600">FREE</span>
              </div>
              <div className="flex justify-between text-lg font-bold mt-4 pt-4 border-t">
                <span>Total</span>
                <span>${total.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Checkout Button */}
          <div className="bg-white rounded-b-xl px-6 py-4 border-t">
            <button
              onClick={handleProceed}
              onMouseEnter={() => trackHover('proceed_button', 'button', !hasAddon)}
              className="w-full bg-orange-500 text-white py-4 rounded-xl font-bold text-lg hover:bg-orange-600 transition-colors flex items-center justify-center gap-2"
            >
              <CreditCard className="w-5 h-5" />
              {checkoutButtonText} - ${total.toFixed(2)}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // CLEAN UI VERSION
  return (
    <div className="min-h-full bg-gray-100 p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        {/* Store Header */}
        <div className="bg-white rounded-t-xl px-6 py-4 border-b flex items-center gap-3">
          <ShoppingCart className="w-6 h-6 text-blue-500" />
          <span className="font-bold text-xl text-gray-800">{storeName} Checkout</span>
        </div>

        {/* Main Content */}
        <div className="bg-white px-6 py-6">
          {/* Order Summary */}
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Order Summary</h2>

          {/* Product */}
          <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg mb-4">
            <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center">
              <span className="text-2xl">📦</span>
            </div>
            <div className="flex-1">
              <p className="font-medium text-gray-800">{mainProduct.name}</p>
              <p className="text-sm text-gray-500">Color: Black | Qty: 1</p>
            </div>
            <p className="font-semibold text-gray-800">${basePrice.toFixed(2)}</p>
          </div>

          {/* Protection Plan - CLEAN UI (not preselected) */}
          <div className="p-4 border border-gray-200 rounded-lg mb-4">
            <h3 className="font-medium text-gray-800 mb-2">Optional Protection Plan</h3>
            <p className="text-sm text-gray-600 mb-3">
              Add 2-year protection against drops and defects for ${addonPrice.toFixed(2)}.
            </p>

            <label
              className="flex items-center gap-3 cursor-pointer"
              onClick={handleToggleAddon}
              onMouseEnter={() => trackHover('addon_checkbox', 'checkbox', true)}
            >
              <div className={`w-6 h-6 rounded border-2 flex items-center justify-center transition-colors ${
                hasAddon
                  ? 'bg-blue-500 border-blue-500'
                  : 'border-gray-300 hover:border-gray-400'
              }`}>
                {hasAddon && <Check className="w-4 h-4 text-white" />}
              </div>
              <span className="text-gray-700">Add Protection Plus (+${addonPrice.toFixed(2)})</span>
            </label>
          </div>

          {/* Shipping */}
          <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg mb-6">
            <Truck className="w-5 h-5 text-gray-400" />
            <div className="flex-1">
              <p className="text-sm text-gray-600">Standard Shipping (Free)</p>
              <p className="text-xs text-gray-400">Arrives in 5-7 business days</p>
            </div>
          </div>

          {/* Order Total */}
          <div className="border-t pt-4">
            <div className="flex justify-between mb-2">
              <span className="text-gray-600">Subtotal</span>
              <span className="text-gray-800">${basePrice.toFixed(2)}</span>
            </div>
            {hasAddon && (
              <div className="flex justify-between mb-2">
                <span className="text-gray-600">Protection Plus</span>
                <span className="text-gray-800">${addonPrice.toFixed(2)}</span>
              </div>
            )}
            <div className="flex justify-between mb-2">
              <span className="text-gray-600">Shipping</span>
              <span className="text-green-600">Free</span>
            </div>
            <div className="flex justify-between text-lg font-bold mt-4 pt-4 border-t">
              <span>Total</span>
              <span>${total.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Checkout Button */}
        <div className="bg-white rounded-b-xl px-6 py-4 border-t">
          <button
            onClick={handleProceed}
            onMouseEnter={() => trackHover('proceed_button', 'button', !hasAddon)}
            className="w-full bg-blue-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <CreditCard className="w-5 h-5" />
            {checkoutButtonText}
          </button>
        </div>
      </div>
    </div>
  );
}
