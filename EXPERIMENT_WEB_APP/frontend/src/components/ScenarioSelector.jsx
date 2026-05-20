/**
 * CYPEARL Scenario Selector - Landing Page
 *
 * Allows participants to select which research scenario they're participating in:
 * 1. Phishing Emails - Email security experiment
 * 2. Dark Patterns - Deceptive UI/UX experiment
 * 3. Fake News - Misinformation detection experiment
 */

import React from "react";
import {
  Mail,
  MousePointer2,
  Newspaper,
  ArrowRight,
  Shield,
} from "lucide-react";

const scenarios = [
  {
    id: "phishing",
    name: "Phishing Emails",
    description:
      "Evaluate email messages and identify potential security threats in a simulated corporate email environment.",
    icon: Mail,
    color: "gray",
    bgColor: "bg-gray-50",
    borderColor: "border-gray-200",
    iconColor: "text-gray-600",
    hoverColor: "hover:border-blue-200 hover:bg-blue-50",
    buttonColor: "bg-gray-600 hover:bg-gray-700 ",
    duration: "~25 minutes",
    tasks: "16 emails",
  },
  {
    id: "dark-patterns",
    name: "Dark Patterns",
    description:
      "Interact with various website interfaces and complete online tasks like managing subscriptions and privacy settings.",
    icon: MousePointer2,
    color: "gray",
    bgColor: "bg-gray-50",
    borderColor: "border-gray-200",
    iconColor: "text-gray-600",
    hoverColor: "hover:border-blue-200 hover:bg-blue-50",
    buttonColor: "bg-gray-600 hover:bg-gray-700",
    duration: "~20 minutes",
    tasks: "16 UI tasks",
  },
  {
    id: "fake-news",
    name: "Fake News",
    description:
      "Browse social media posts and news headlines, evaluating their accuracy and deciding whether you would share them.",
    icon: Newspaper,
    color: "gray",
    bgColor: "bg-gray-50",
    borderColor: "border-gray-200",
    iconColor: "text-gray-600",
    hoverColor: "hover:border-blue-200 hover:bg-blue-50",
    buttonColor: "bg-gray-600 hover:bg-gray-700",
    duration: "~20 minutes",
    tasks: "16 news items",
  },
];

export default function ScenarioSelector({ onSelectScenario }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-4">
      <div className="max-w-5xl w-full">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-slate-800 rounded-xl flex items-center justify-center">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-slate-800">CYPEARL</h1>
          </div>
          <p className="text-lg text-slate-600 mb-2">
            Cyber Synthetic Persona Evaluation for Adversarial Resilience Lab
          </p>
          <p className="text-slate-500 max-w-2xl mx-auto">
            Welcome to our research study. Please select the experiment you've
            been assigned to participate in.
          </p>
        </div>

        {/* Scenario Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {scenarios.map((scenario) => {
            const Icon = scenario.icon;
            return (
              <div
                key={scenario.id}
                className={`${scenario.bgColor} ${scenario.borderColor} ${scenario.hoverColor} border-2 rounded-2xl p-6 transition-all duration-300 cursor-pointer group hover:shadow-lg`}
                onClick={() => onSelectScenario(scenario.id)}
              >
                {/* Icon */}
                <div
                  className={`w-14 h-14 rounded-xl ${scenario.bgColor} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
                >
                  <Icon className={`w-8 h-8 ${scenario.iconColor}`} />
                </div>

                {/* Title */}
                <h2 className="text-xl font-bold text-slate-800 mb-2">
                  {scenario.name}
                </h2>

                {/* Description */}
                <p className="text-slate-600 text-sm mb-4 leading-relaxed">
                  {scenario.description}
                </p>

                {/* Meta Info */}
                <div className="flex items-center gap-4 text-xs text-slate-500 mb-4">
                  <span className="bg-white/60 px-2 py-1 rounded-full">
                    {scenario.duration}
                  </span>
                  <span className="bg-white/60 px-2 py-1 rounded-full">
                    {scenario.tasks}
                  </span>
                </div>

                {/* Select Button */}
                <button
                  className={`w-full ${scenario.buttonColor} text-white py-3 px-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors group-hover:gap-3`}
                >
                  Start Experiment
                  <ArrowRight
                    size={18}
                    className="transition-transform group-hover:translate-x-1"
                  />
                </button>
              </div>
            );
          })}
        </div>

        {/* Footer Note */}
        <div className="text-center text-sm text-slate-500">
          <p>
            If you're unsure which experiment to select, please check your
            Prolific study instructions.
          </p>
        </div>
      </div>
    </div>
  );
}
