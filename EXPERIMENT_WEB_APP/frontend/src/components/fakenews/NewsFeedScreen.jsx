/**
 * NewsFeedScreen - Twitter/X style tweet display with evaluation (Light Mode)
 *
 * Displays a news item as a Twitter/X tweet format
 * and collects user evaluations (accuracy rating, sharing intention, etc.)
 *
 * Tracks behavioral metrics:
 * - Reading time
 * - Source hover/click
 * - Scroll depth
 * - Engagement button hover
 */

import React, { useState, useEffect, useRef } from "react";
import {
  MessageCircle,
  Repeat2,
  Heart,
  Bookmark,
  Share,
  MoreHorizontal,
  CheckCircle,
  ArrowRight,
  Play,
} from "lucide-react";

// Twitter/X Verification Badge Component with Popup
const VerificationBadge = ({ type, showPopup, onBadgeClick, onClosePopup }) => {
  if (type === "none") return null;

  const badgeColors = {
    gold: "#F5C33B",
    blue: "#1D9BF0",
    grey: "#6E7A89",
  };

  const badgeInfo = {
    gold: {
      title: "Verified account",
      description:
        "This account is verified because it's an official organization on X.",
      learnMore: "Learn more",
      verifiedSince: "Verified since October 2021.",
    },
    blue: {
      title: "Verified account",
      description:
        "This account is verified because they are subscribed to X Premium.",
      learnMore: "Learn more",
      verifiedSince: "Verified since February 2018.",
    },
    grey: {
      title: "Verified account",
      description:
        "This account is verified because it is a government or multilateral organization account.",
      learnMore: "Learn more",
      verifiedSince: "Verified since June 2010.",
    },
  };

  const info = badgeInfo[type];

  return (
    <div className="relative">
      <div
        className="w-[22px] h-[18px] flex items-center justify-center ml-0.5 flex-shrink-0 cursor-pointer"
        onClick={(e) => {
          e.stopPropagation();
          onBadgeClick?.();
        }}
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M10.5213 2.62368C11.3147 1.75255 12.6853 1.75255 13.4787 2.62368L14.4989 3.74391C14.8998 4.18418 15.4761 4.42288 16.071 4.39508L17.5845 4.32435C18.7614 4.26934 19.7307 5.23857 19.6757 6.41554L19.6049 7.92905C19.5771 8.52388 19.8158 9.10016 20.2561 9.50111L21.3763 10.5213C22.2475 11.3147 22.2475 12.6853 21.3763 13.4787L20.2561 14.4989C19.8158 14.8998 19.5771 15.4761 19.6049 16.071L19.6757 17.5845C19.7307 18.7614 18.7614 19.7307 17.5845 19.6757L16.071 19.6049C15.4761 19.5771 14.8998 19.8158 14.4989 20.2561L13.4787 21.3763C12.6853 22.2475 11.3147 22.2475 10.5213 21.3763L9.50111 20.2561C9.10016 19.8158 8.52388 19.5771 7.92905 19.6049L6.41553 19.6757C5.23857 19.7307 4.26934 18.7614 4.32435 17.5845L4.39508 16.071C4.42288 15.4761 4.18418 14.8998 3.74391 14.4989L2.62368 13.4787C1.75255 12.6853 1.75255 11.3147 2.62368 10.5213L3.74391 9.50111C4.18418 9.10016 4.42288 8.52388 4.39508 7.92905L4.32435 6.41553C4.26934 5.23857 5.23857 4.26934 6.41554 4.32435L7.92905 4.39508C8.52388 4.42288 9.10016 4.18418 9.50111 3.74391L10.5213 2.62368Z"
            fill={badgeColors[type]}
          />
          <path
            d="M9 12L11 14L15 10"
            fill="none"
            stroke="white"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>

      {/* Verification Popup */}
      {showPopup && (
        <>
          {/* Backdrop to close popup */}
          <div
            className="fixed inset-0 z-40"
            onClick={(e) => {
              e.stopPropagation();
              onClosePopup?.();
            }}
          />
          {/* Popup card */}
          <div className="absolute left-0 top-6 z-50 w-72 bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
            {/* Header with badge icon */}
            <div className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M10.5213 2.62368C11.3147 1.75255 12.6853 1.75255 13.4787 2.62368L14.4989 3.74391C14.8998 4.18418 15.4761 4.42288 16.071 4.39508L17.5845 4.32435C18.7614 4.26934 19.7307 5.23857 19.6757 6.41554L19.6049 7.92905C19.5771 8.52388 19.8158 9.10016 20.2561 9.50111L21.3763 10.5213C22.2475 11.3147 22.2475 12.6853 21.3763 13.4787L20.2561 14.4989C19.8158 14.8998 19.5771 15.4761 19.6049 16.071L19.6757 17.5845C19.7307 18.7614 18.7614 19.7307 17.5845 19.6757L16.071 19.6049C15.4761 19.5771 14.8998 19.8158 14.4989 20.2561L13.4787 21.3763C12.6853 22.2475 11.3147 22.2475 10.5213 21.3763L9.50111 20.2561C9.10016 19.8158 8.52388 19.5771 7.92905 19.6049L6.41553 19.6757C5.23857 19.7307 4.26934 18.7614 4.32435 17.5845L4.39508 16.071C4.42288 15.4761 4.18418 14.8998 3.74391 14.4989L2.62368 13.4787C1.75255 12.6853 1.75255 11.3147 2.62368 10.5213L3.74391 9.50111C4.18418 9.10016 4.42288 8.52388 4.39508 7.92905L4.32435 6.41553C4.26934 5.23857 5.23857 4.26934 6.41554 4.32435L7.92905 4.39508C8.52388 4.42288 9.10016 4.18418 9.50111 3.74391L10.5213 2.62368Z"
                    fill={badgeColors[type]}
                  />
                  <path
                    d="M9 12L11 14L15 10"
                    fill="none"
                    stroke="white"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <span className="font-bold text-gray-900">{info.title}</span>
              </div>
              <p className="text-gray-600 text-sm leading-relaxed">
                {info.description}{" "}
                <span className="text-sky-500 hover:underline cursor-pointer">
                  {info.learnMore}
                </span>
              </p>
            </div>
            {/* Divider */}
            <div className="border-t border-gray-100" />
            {/* Verified since */}
            <div className="px-4 py-3">
              <p className="text-gray-500 text-sm">{info.verifiedSince}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// Likert scale component
const LikertScale = ({
  value,
  onChange,
  leftLabel,
  rightLabel,
  questionText,
}) => {
  return (
    <div className="space-y-3">
      <p className="font-medium text-gray-800">{questionText}</p>
      <div className="flex items-center justify-between gap-2">
        {[1, 2, 3, 4, 5, 6, 7].map((n) => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className={`w-10 h-10 rounded-full text-sm font-semibold transition-all
              ${
                value === n
                  ? "bg-sky-500 text-white scale-110 shadow-lg"
                  : "bg-gray-100 text-gray-600 hover:bg-sky-100 hover:text-sky-700"
              }`}
          >
            {n}
          </button>
        ))}
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  );
};

export default function NewsFeedScreen({
  newsItem,
  onComplete,
  questionOrder,
  progress,
}) {
  // Evaluation state
  const [accuracyRating, setAccuracyRating] = useState(null);
  const [sharingIntention, setSharingIntention] = useState(null);
  const [seenBefore, setSeenBefore] = useState(null);
  const [confidence, setConfidence] = useState(null);

  // Qualitative response state (for chain-of-thought LLM conditioning)
  const [cuesNoticed, setCuesNoticed] = useState("");
  const [evaluationProcess, setEvaluationProcess] = useState("");
  const [influencingFactors, setInfluencingFactors] = useState("");
  const [uncertaintyPoints, setUncertaintyPoints] = useState("");

  // Badge popup state
  const [showBadgePopup, setShowBadgePopup] = useState(false);

  // Metrics tracking
  const [metrics, setMetrics] = useState({
    displayTime: Date.now(),
    sourceHoverStart: null,
    sourceHoverTotal: 0,
    sourceClicked: false,
    engagementHovered: false,
    scrollEvents: [],
    maxScrollDepth: 0,
    headlineRevisits: 0,
    accuracyAnsweredAt: null,
    sharingAnsweredAt: null,
    badgeClicked: false,
  });

  // Refs
  const containerRef = useRef(null);
  const tweetRef = useRef(null);

  // Track scroll events
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let lastScrollTop = 0;

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const scrollHeight = container.scrollHeight - container.clientHeight;
      const depth = scrollHeight > 0 ? scrollTop / scrollHeight : 0;

      // Detect scroll back to tweet (top)
      if (
        depth < 0.1 &&
        metrics.maxScrollDepth > 0.3 &&
        scrollTop < lastScrollTop
      ) {
        setMetrics((prev) => ({
          ...prev,
          headlineRevisits: prev.headlineRevisits + 1,
        }));
      }

      setMetrics((prev) => ({
        ...prev,
        maxScrollDepth: Math.max(prev.maxScrollDepth, depth),
        scrollEvents: [...prev.scrollEvents, { depth, timestamp: Date.now() }],
      }));

      lastScrollTop = scrollTop;
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [metrics.maxScrollDepth]);

  // Source hover tracking
  const handleSourceHover = (isEntering) => {
    if (isEntering) {
      setMetrics((prev) => ({ ...prev, sourceHoverStart: Date.now() }));
    } else if (metrics.sourceHoverStart) {
      const hoverDuration = Date.now() - metrics.sourceHoverStart;
      setMetrics((prev) => ({
        ...prev,
        sourceHoverTotal: prev.sourceHoverTotal + hoverDuration,
        sourceHoverStart: null,
      }));
    }
  };

  // Source click tracking
  const handleSourceClick = () => {
    setMetrics((prev) => ({ ...prev, sourceClicked: true }));
  };

  // Engagement button hover tracking
  const handleEngagementHover = () => {
    setMetrics((prev) => ({ ...prev, engagementHovered: true }));
  };

  // Track accuracy answer time
  const handleAccuracyChange = (value) => {
    setAccuracyRating(value);
    if (!metrics.accuracyAnsweredAt) {
      setMetrics((prev) => ({
        ...prev,
        accuracyAnsweredAt: Date.now() - prev.displayTime,
      }));
    }
  };

  // Track sharing answer time
  const handleSharingChange = (value) => {
    setSharingIntention(value);
    if (!metrics.sharingAnsweredAt) {
      setMetrics((prev) => ({
        ...prev,
        sharingAnsweredAt: Date.now() - prev.displayTime,
      }));
    }
  };

  // Submit evaluation
  const handleSubmit = () => {
    const readingTime = Date.now() - metrics.displayTime;

    // Determine time to first answer based on question order
    let timeToAccuracy, timeToSharing;
    if (questionOrder === "accuracy_first") {
      timeToAccuracy = metrics.accuracyAnsweredAt || readingTime;
      timeToSharing = metrics.sharingAnsweredAt || readingTime;
    } else {
      timeToSharing = metrics.sharingAnsweredAt || readingTime;
      timeToAccuracy = metrics.accuracyAnsweredAt || readingTime;
    }

    // Combine qualitative responses into structured format (like phishing experiment)
    const qualitativeReason = [
      `CUES_NOTICED: ${cuesNoticed.trim() || "N/A"}`,
      `EVALUATION_PROCESS: ${evaluationProcess.trim() || "N/A"}`,
      `INFLUENCING_FACTORS: ${influencingFactors.trim() || "N/A"}`,
      `UNCERTAINTY_POINTS: ${uncertaintyPoints.trim() || "N/A"}`,
    ].join("\n");

    onComplete({
      item_id: newsItem.item_id,
      accuracy_rating: accuracyRating,
      sharing_intention: sharingIntention,
      seen_before: seenBefore,
      confidence: confidence,
      // Qualitative responses
      reason: qualitativeReason,
      cues_noticed: cuesNoticed.trim(),
      evaluation_process: evaluationProcess.trim(),
      influencing_factors: influencingFactors.trim(),
      uncertainty_points: uncertaintyPoints.trim(),
      // Timing
      reading_time_ms: readingTime,
      time_to_accuracy_judgment_ms: timeToAccuracy,
      time_to_sharing_judgment_ms: timeToSharing,
      source_hover: metrics.sourceHoverTotal > 0,
      source_hover_time_ms: metrics.sourceHoverTotal,
      source_click: metrics.sourceClicked,
      badge_click: metrics.badgeClicked,
      headline_reread: metrics.headlineRevisits > 0,
      engagement_hover: metrics.engagementHovered,
      scroll_depth: metrics.maxScrollDepth,
      scroll_events: metrics.scrollEvents.slice(-50), // Limit to last 50 events
      hover_events: [],
    });
  };

  const canSubmit = true; /* TODO: re-enable validation for production:
    accuracyRating !== null &&
    sharingIntention !== null &&
    seenBefore !== null  && */
  confidence !== null;

  // Format engagement numbers
  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
    if (num >= 1000) return (num / 1000).toFixed(1) + "K";
    return num.toString();
  };

  // Render questions based on order
  const renderQuestions = () => {
    const accuracyQuestion = (
      <LikertScale
        key="accuracy"
        value={accuracyRating}
        onChange={handleAccuracyChange}
        questionText="How accurate do you think this post is?"
        leftLabel="Definitely False"
        rightLabel="Definitely True"
      />
    );

    const sharingQuestion = (
      <LikertScale
        key="sharing"
        value={sharingIntention}
        onChange={handleSharingChange}
        questionText="If you saw this on social media, how likely would you be to share/repost it?"
        leftLabel="Definitely Would Not Share"
        rightLabel="Definitely Would Share"
      />
    );

    if (questionOrder === "accuracy_first") {
      return (
        <>
          {accuracyQuestion}
          <div className="border-t pt-6">{sharingQuestion}</div>
        </>
      );
    } else {
      return (
        <>
          {sharingQuestion}
          <div className="border-t pt-6">{accuracyQuestion}</div>
        </>
      );
    }
  };

  // Parse tweet content to highlight hashtags
  const renderTweetContent = (content) => {
    const parts = content.split(/(#\w+)/g);
    return parts.map((part, i) => {
      if (part.startsWith("#")) {
        return (
          <span key={i} className="text-sky-500 hover:underline cursor-pointer">
            {part}
          </span>
        );
      }
      return part;
    });
  };

  // Render media (image or video)
  const renderMedia = () => {
    if (!newsItem.thumbnail) return null;

    if (newsItem.thumbnail_type === "video") {
      return (
        <div className="mt-3 rounded-2xl overflow-hidden border border-gray-200 relative">
          <video
            src={newsItem.thumbnail}
            className="w-full max-h-80 object-cover bg-gray-100"
            autoPlay
            muted
            loop
            playsInline
            onError={(e) => {
              e.target.style.display = "none";
              e.target.nextSibling.style.display = "flex";
            }}
          />
          {/* Fallback for video */}
          <div className="w-full h-48 bg-gradient-to-br from-gray-100 to-gray-200 items-center justify-center hidden">
            <div className="w-16 h-16 bg-gray-300 rounded-full flex items-center justify-center">
              <Play className="w-8 h-8 text-gray-500 ml-1" />
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-3 rounded-2xl overflow-hidden border border-gray-200">
        <img
          src={newsItem.thumbnail}
          alt="Post media"
          className="w-full max-h-80 object-cover bg-gray-100"
          onError={(e) => {
            e.target.style.display = "none";
            e.target.nextSibling.style.display = "flex";
          }}
        />
        {/* Fallback for image */}
        <div className="w-full h-48 bg-gradient-to-br from-gray-100 to-gray-200 items-center justify-center hidden">
          <span className="text-gray-400 text-sm">Image unavailable</span>
        </div>
      </div>
    );
  };

  return (
    <div
      ref={containerRef}
      className="min-h-screen bg-gray-50 overflow-y-auto pb-8"
    >
      <div className="max-w-2xl mx-auto pt-4 px-4">
        {/* Twitter/X Post Card - Light Mode */}
        <div className="bg-white border border-gray-800 rounded-2xl pr-8">
          {/* Post Header */}
          <div className="px-4 pt-3">
            <div className="flex items-start gap-3">
              {/* Avatar/Logo */}
              <div
                className="w-10 h-10 rounded-full overflow-hidden cursor-pointer flex-shrink-0 bg-gray-100"
                onMouseEnter={() => handleSourceHover(true)}
                onMouseLeave={() => handleSourceHover(false)}
                onClick={handleSourceClick}
              >
                {newsItem.source_logo ? (
                  <img
                    src={newsItem.source_logo}
                    alt={newsItem.display_name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.style.display = "none";
                      e.target.nextSibling.style.display = "flex";
                    }}
                  />
                ) : null}
                {/* Fallback avatar with initials */}
                <div
                  className="w-full h-full bg-sky-500 items-center justify-center"
                  style={{ display: newsItem.source_logo ? "none" : "flex" }}
                >
                  <span className="text-white font-bold text-sm">
                    {newsItem.display_name
                      .split(" ")
                      .map((w) => w[0])
                      .join("")
                      .slice(0, 2)
                      .toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                {/* User Info Row */}
                <div className="flex items-center gap-1">
                  <div
                    className="flex items-center gap-1 cursor-pointer min-w-0"
                    onMouseEnter={() => handleSourceHover(true)}
                    onMouseLeave={() => handleSourceHover(false)}
                    onClick={handleSourceClick}
                  >
                    <span className="font-bold text-gray-900 hover:underline truncate text-[15px]">
                      {newsItem.display_name}
                    </span>
                    <VerificationBadge
                      type={newsItem.badge_type}
                      showPopup={showBadgePopup}
                      onBadgeClick={() => {
                        setShowBadgePopup(true);
                        setMetrics((prev) => ({ ...prev, badgeClicked: true }));
                      }}
                      onClosePopup={() => setShowBadgePopup(false)}
                    />
                  </div>
                  <span className="text-gray-500 truncate text-[15px]">
                    {newsItem.handle}
                  </span>
                  <span className="text-gray-500 text-[15px]">·</span>
                  <span className="text-gray-500 hover:underline cursor-pointer text-[15px]">
                    2h
                  </span>
                  <button className="ml-auto p-2 hover:bg-gray-100 rounded-full -mr-2 -mt-1">
                    <MoreHorizontal className="w-5 h-5 text-gray-500" />
                  </button>
                </div>

                {/* Tweet Content */}
                <div ref={tweetRef} className="mt-0.5">
                  <p className="text-gray-900 text-[15px] leading-normal whitespace-pre-wrap">
                    {renderTweetContent(newsItem.tweet_content)}
                  </p>
                </div>

                {/* Media (Image/Video) */}
                {renderMedia()}

                {/* Engagement Stats */}
                <div
                  className="flex items-center justify-between mt-3 max-w-md"
                  onMouseEnter={handleEngagementHover}
                >
                  {/* Reply */}
                  <button className="flex items-center gap-1 group">
                    <div className="p-2 rounded-full group-hover:bg-sky-50 transition-colors">
                      <MessageCircle className="w-[18px] h-[18px] text-gray-500 group-hover:text-sky-500" />
                    </div>
                    <span className="text-[13px] text-gray-500 group-hover:text-sky-500">
                      {formatNumber(newsItem.engagement_counts?.replies || 0)}
                    </span>
                  </button>

                  {/* Repost */}
                  <button className="flex items-center gap-1 group">
                    <div className="p-2 rounded-full group-hover:bg-green-50 transition-colors">
                      <Repeat2 className="w-[18px] h-[18px] text-gray-500 group-hover:text-green-600" />
                    </div>
                    <span className="text-[13px] text-gray-500 group-hover:text-green-600">
                      {formatNumber(newsItem.engagement_counts?.retweets || 0)}
                    </span>
                  </button>

                  {/* Like */}
                  <button className="flex items-center gap-1 group">
                    <div className="p-2 rounded-full group-hover:bg-pink-50 transition-colors">
                      <Heart className="w-[18px] h-[18px] text-gray-500 group-hover:text-pink-600" />
                    </div>
                    <span className="text-[13px] text-gray-500 group-hover:text-pink-600">
                      {formatNumber(newsItem.engagement_counts?.likes || 0)}
                    </span>
                  </button>

                  {/* Views */}
                  <button className="flex items-center gap-1 group">
                    <div className="p-2 rounded-full group-hover:bg-sky-50 transition-colors">
                      <svg
                        viewBox="0 0 24 24"
                        className="w-[18px] h-[18px] text-gray-500 group-hover:text-sky-500 fill-current"
                      >
                        <path d="M8.75 21V3h2v18h-2zM18 21V8.5h2V21h-2zM4 21l.004-10h2L6 21H4zm9.248 0v-7h2v7h-2z" />
                      </svg>
                    </div>
                    <span className="text-[13px] text-gray-500 group-hover:text-sky-500">
                      {formatNumber(newsItem.engagement_counts?.views || 0)}
                    </span>
                  </button>

                  {/* Bookmark & Share */}
                  <div className="flex items-center">
                    <button className="p-2 rounded-full hover:bg-sky-50 transition-colors group">
                      <Bookmark className="w-[18px] h-[18px] text-gray-500 group-hover:text-sky-500" />
                    </button>
                    <button className="p-2 rounded-full hover:bg-sky-50 transition-colors group">
                      <Share className="w-[18px] h-[18px] text-gray-500 group-hover:text-sky-500" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom padding */}
          <div className="h-3"></div>
        </div>

        {/* Evaluation Questions */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 space-y-6 mt-4">
          <div className="flex items-center gap-2 text-sky-600 mb-4">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">Your Evaluation</span>
          </div>

          {/* Main Questions (counterbalanced order) */}
          {renderQuestions()}

          {/* Seen Before */}
          <div className="border-t pt-6 space-y-3">
            <p className="font-medium text-gray-800">
              Have you seen this post before?
            </p>
            <div className="flex gap-4">
              <button
                onClick={() => setSeenBefore(true)}
                className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                  seenBefore === true
                    ? "bg-sky-500 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                Yes
              </button>
              <button
                onClick={() => setSeenBefore(false)}
                className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                  seenBefore === false
                    ? "bg-sky-500 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                No
              </button>
            </div>
          </div>

          {/* Confidence */}
          <div className="border-t pt-6">
            <LikertScale
              value={confidence}
              onChange={setConfidence}
              questionText="How confident are you in your accuracy rating?"
              leftLabel="Not at all confident"
              rightLabel="Extremely confident"
            />
          </div>

          {/* Qualitative Responses Section */}
          <div className="border-t pt-6 space-y-5">
            <div className="flex items-center gap-2 text-sky-600 mb-2">
              <span className="font-semibold">Explain Your Evaluation</span>
            </div>
            <p className="text-sm text-gray-500 -mt-2">
              Please briefly explain how you evaluated this post. Your responses
              will help us understand human reasoning.
            </p>

            {/* Cues Noticed */}
            <div className="space-y-2">
              <label className="block font-medium text-gray-800 text-sm">
                What specific cues or details helped you evaluate this post?
              </label>
              <textarea
                value={cuesNoticed}
                onChange={(e) => setCuesNoticed(e.target.value)}
                placeholder="e.g., the account name, verification badge, wording, hashtags, etc."
                className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-all"
                rows={2}
              />
            </div>

            {/* Evaluation Process */}
            <div className="space-y-2">
              <label className="block font-medium text-gray-800 text-sm">
                How did you decide whether this post was accurate?
              </label>
              <textarea
                value={evaluationProcess}
                onChange={(e) => setEvaluationProcess(e.target.value)}
                placeholder="e.g., I checked if the account seemed trustworthy, I considered if the claim was plausible..."
                className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-all"
                rows={2}
              />
            </div>

            {/* Influencing Factors */}
            <div className="space-y-2">
              <label className="block font-medium text-gray-800 text-sm">
                What factors most influenced your judgment?
              </label>
              <textarea
                value={influencingFactors}
                onChange={(e) => setInfluencingFactors(e.target.value)}
                placeholder="e.g., verification status, emotional language, topic familiarity, etc."
                className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-all"
                rows={2}
              />
            </div>

            {/* Uncertainty Points */}
            <div className="space-y-2">
              <label className="block font-medium text-gray-800 text-sm">
                What were you unsure about? (if anything)
              </label>
              <textarea
                value={uncertaintyPoints}
                onChange={(e) => setUncertaintyPoints(e.target.value)}
                placeholder="e.g., I wasn't sure if this account is real, I couldn't verify the claim..."
                className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-all"
                rows={2}
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="pt-4">
            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
                canSubmit
                  ? "bg-sky-500 hover:bg-sky-600 text-white"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              Continue to Next Item
              <ArrowRight className="w-5 h-5" />
            </button>
            {!canSubmit && (
              <p className="text-center text-sm text-gray-500 mt-2">
                Please answer all questions to continue
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
